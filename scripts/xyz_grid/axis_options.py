import os
from pprint import pprint

import gradio as gr
import modules
import modules.scripts as scripts
import modules.shared as shared
from modules import sd_samplers
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.sd_models import model_data
from modules.sd_models import select_checkpoint
from modules.shared import opts


def format_value_add_label(p, opt, x):
    if isinstance(x, float):
        x = round(x, 8)

    return f"{opt.label}: {x}"


def boolean_choice(reverse: bool = False):
    def choice():
        return ["False", "True"] if reverse else ["True", "False"]

    return choice


def refresh_loading_params_for_xyz_grid():
    """
    Refreshes the loading parameters for the model,
    prompts a reload in sd_models.forge_model_reload()
    """
    checkpoint_info = select_checkpoint()

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        # unet_storage_dtype=shared.opts.forge_unet_storage_dtype
        unet_storage_dtype=model_data.forge_loading_parameters.get("unet_storage_dtype", None),
    )


def set_script_arg(p, script, idx, value):
    if not hasattr(p, "script_args") or p.script_args is None:
        p.script_args = []
    p.script_args = list(p.script_args)  # Ensure mutable
    arg_pos = script.args_from + idx
    while len(p.script_args) <= arg_pos:
        p.script_args.append(None)
    p.script_args[arg_pos] = value


def guess_type_from_component(component):
    if isinstance(component, gr.Checkbox):
        return bool
    elif isinstance(component, gr.Slider):
        return float
    elif isinstance(component, gr.Dropdown):
        return str
    else:
        return str  # default for other components


class AxisOption:
    def __init__(
        self,
        label,
        type,
        apply,
        format_value=format_value_add_label,
        confirm=None,
        cost=0.0,
        choices=None,
        prepare=None,
    ):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value
        self.confirm = confirm
        self.cost = cost
        self.prepare = prepare
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True


class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    token_order = []

    # Initially grab the tokens from the prompt, so they can be replaced in
    # order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token) :]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            raise RuntimeError(f"Unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    # skip if the checkpoint was last override
    if info.name == p.override_settings.get("sd_model_checkpoint", None):
        return
    org_cp = getattr(opts, "sd_model_checkpoint", None)
    p.override_settings["sd_model_checkpoint"] = info.name
    opts.set("sd_model_checkpoint", info.name)
    refresh_loading_params_for_xyz_grid()
    # This saves part of the reload
    opts.set("sd_model_checkpoint", org_cp)


def confirm_checkpoints(p, xs):
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def confirm_checkpoints_or_none(p, xs):
    for x in xs:
        if x in (None, "", "None", "none"):
            continue

        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def confirm_range(min_val, max_val, axis_label):
    """Generates a AxisOption.confirm() function that checks all values are within the specified range."""

    def confirm_range_fun(p, xs):
        for x in xs:
            if not (max_val >= x >= min_val):
                raise ValueError(f'{axis_label} value "{x}" out of range [{min_val}, {max_val}]')

    return confirm_range_fun


def apply_size(p, x: str, xs) -> None:
    try:
        width, _, height = x.partition("x")
        width = int(width.strip())
        height = int(height.strip())
        p.width = width
        p.height = height
    except ValueError:
        print(f"Invalid size in XYZ plot: {x}")


def apply_vae(p, x, xs):
    p.override_settings["sd_vae"] = find_vae(x)


def apply_styles(p: StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles.extend(x.split(","))


def apply_uni_pc_order(p, x, xs):
    p.override_settings["uni_pc_order"] = min(x, p.steps - 1)


def apply_face_restore(p, opt, x):
    opt = opt.lower()
    if opt == "codeformer":
        is_active = True
        p.face_restoration_model = "CodeFormer"
    elif opt == "gfpgan":
        is_active = True
        p.face_restoration_model = "GFPGAN"
    else:
        is_active = opt in ("true", "yes", "y", "1")

    p.restore_faces = is_active


def apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        p.override_settings[field] = x

    return fun


def format_value(p, opt, x):
    if isinstance(x, float):
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def format_remove_path(p, opt, x):
    return os.path.basename(x)


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


def find_vae(name: str):
    if (name := name.strip().lower()) in ("auto", "automatic"):
        return "Automatic"
    elif name == "none":
        return "None"
    return next(
        (k for k in modules.sd_vae.vae_dict if k.lower() == name),
        print(f"No VAE found for {name}; using Automatic") or "Automatic",
    )


axis_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    AxisOption("Seed", int, apply_field("seed")),
          AxisOption("Var. seed", int, apply_field("subseed")),
    #    #    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    #    AxisOption("Steps", int, apply_field("steps")),
    #    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    #    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    #    #    AxisOption("Distilled CFG Scale", float, apply_field("distilled_cfg_scale")),
    #    #    AxisOptionImg2Img("Image CFG Scale", float, apply_field("image_cfg_scale")),
       AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
       AxisOption(
           "Prompt order",
           str_permutations,
           apply_order,
           format_value=format_value_join_list,
       ),
       AxisOptionTxt2Img(
           "Sampler",
           str,
           apply_field("sampler_name"),
           format_value=format_value,
           confirm=confirm_samplers,
           choices=lambda: [x.name for x in sd_samplers.samplers if x.name not in opts.hide_samplers],
       ),
    #    #    AxisOptionTxt2Img("Hires sampler", str, apply_field("hr_sampler_name"), confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    #    AxisOptionImg2Img(
    #        "Sampler",
    #        str,
    #        apply_field("sampler_name"),
    #        format_value=format_value,
    #        confirm=confirm_samplers,
    #        choices=lambda: [
    #            x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers
    #        ],
    #    ),
       AxisOption(
           "Checkpoint name",
           str,
           apply_checkpoint,
           format_value=format_remove_path,
           confirm=confirm_checkpoints,
           cost=1.0,
           choices=lambda: sorted(sd_models.checkpoints_list, key=str.casefold),
       ),
    #    AxisOption("Negative Guidance minimum sigma", float, apply_field("s_min_uncond")),
    #    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    #    AxisOption("Sigma min", float, apply_field("s_tmin")),
    #    AxisOption("Sigma max", float, apply_field("s_tmax")),
    #    AxisOption("Sigma noise", float, apply_field("s_noise")),
    #    AxisOption("Schedule type", str, apply_field("scheduler"), choices=lambda: [x.label for x in sd_schedulers.schedulers]),
    #    AxisOption("Schedule min sigma", float, apply_override("sigma_min")),
    #    AxisOption("Schedule max sigma", float, apply_override("sigma_max")),
    #    AxisOption("Schedule rho", float, apply_override("rho")),
    #    AxisOption("Beta schedule alpha", float, apply_override("beta_dist_alpha")),
    #    AxisOption("Beta schedule beta", float, apply_override("beta_dist_beta")),
    #    AxisOption("Eta", float, apply_field("eta")),
    #    AxisOption("Clip skip", int, apply_override('CLIP_stop_at_last_layers')),
       AxisOption("Denoising", float, apply_field("denoising_strength")),
    #    AxisOption("Initial noise multiplier", float, apply_field("initial_noise_multiplier")),
    #    AxisOption("Extra noise", float, apply_override("img2img_extra_noise")),
    #    AxisOptionTxt2Img("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    #    AxisOptionImg2Img("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight")),
    #    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ['Automatic', 'None'] + list(sd_vae.vae_dict)),
    #    AxisOption("Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)),
    #    AxisOption("UniPC Order", int, apply_uni_pc_order, cost=0.5),
    #    AxisOption("Face restore", str, apply_face_restore, format_value=format_value),
    #    AxisOption("Token merging ratio", float, apply_override('token_merging_ratio')),
    #    AxisOption("Token merging ratio high-res", float, apply_override('token_merging_ratio_hr')),
    #    AxisOption("Always discard next-to-last sigma", str, apply_override('always_discard_next_to_last_sigma', boolean=True), choices=boolean_choice(reverse=True)),
    #    AxisOption("SGM noise multiplier", str, apply_override('sgm_noise_multiplier', boolean=True), choices=boolean_choice(reverse=True)),
    #    AxisOption("Refiner checkpoint", str, apply_field('refiner_checkpoint'), format_value=format_remove_path, confirm=confirm_checkpoints_or_none, cost=1.0, choices=lambda: ['None'] + sorted(sd_models.checkpoints_list, key=str.casefold)),
    #    AxisOption("Refiner switch at", float, apply_field('refiner_switch_at')),
    #    AxisOption("RNG source", str, apply_override("randn_source"), choices=lambda: ["GPU", "CPU", "NV"]),
    #    AxisOption("FP8 mode", str, apply_override("fp8_storage"), cost=0.9, choices=lambda: ["Disable", "Enable for SDXL", "Enable"]),
    #    AxisOption("Size", str, apply_size),
]


def script_axis_options():
    for script in scripts.scripts_txt2img.scripts:
        if not script.infotext_fields:
            continue
        for idx, field in enumerate(script.infotext_fields):
            comp, label = field
            if script.title():
                label = script.title() + "-" + label
            pprint(label)
            if isinstance(comp, gr.Slider):
                opt = AxisOption(
                    label=label,
                    type=float,
                    apply=apply_field(comp.label),
                    format_value=format_value,
                )
            elif isinstance(comp, gr.Number):
                opt = AxisOption(
                    label=label,
                    type=int,
                    apply=apply_field(comp.label),
                )
            elif isinstance(comp, gr.Dropdown):
                opt = AxisOption(
                    label=label,
                    type=str,
                    apply=apply_field(comp.label),
                    choices=lambda: comp.choices if isinstance(comp.choices, list) else [],
                )
            elif isinstance(comp, gr.Checkbox):
                opt = AxisOption(
                    label=label,
                    type=bool,
                    apply=apply_override(comp.label, boolean=True),
                )
            else:
                opt = AxisOption(
                    label=label,
                    type=str,
                    apply=apply_field(comp.label),
                )
            axis_options.append(opt)