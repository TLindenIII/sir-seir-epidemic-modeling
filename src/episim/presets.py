from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresetCitation:
    label: str
    url: str


@dataclass(frozen=True, slots=True)
class DiseasePreset:
    key: str
    disease: str
    model_hint: str
    beta: float
    gamma: float
    sigma: float | None
    note: str
    citations: tuple[PresetCitation, ...]


DISEASE_PRESETS = (
    DiseasePreset(
        key="covid19_ancestral",
        disease="COVID-19 (ancestral strain)",
        model_hint="SEIR-friendly preset",
        beta=0.102,
        gamma=0.050,
        sigma=0.090,
        note="Derived for the simple SEIR model from R0 = 2.03, progression rate ω = 0.09 day^-1, and symptomatic recovery rate γS = 0.05 day^-1 reported by Mwalili et al. (2020).",
        citations=(
            PresetCitation(
                label="Mwalili et al. 2020, BMC Research Notes",
                url="https://link.springer.com/article/10.1186/s13104-020-05192-1",
            ),
        ),
    ),
    DiseasePreset(
        key="influenza_h1n1",
        disease="Influenza A(H1N1) 2009",
        model_hint="SEIR-friendly preset",
        beta=0.311,
        gamma=0.217,
        sigma=0.699,
        note="Derived for a simple SEIR model from R0 = 1.43 for H1N1 2009, an incubation period of 1.43 days, and an infectious duration of 4.6 days as synthesized in McSwiggan et al. (2023).",
        citations=(
            PresetCitation(
                label="McSwiggan et al. 2023, influenza parameter review",
                url="https://usher.ed.ac.uk/sites/default/files/atoms/files/rr_of_influenza_transmission_parameters_v2_formatted.pdf",
            ),
        ),
    ),
    DiseasePreset(
        key="ebola_guinea_2014",
        disease="Ebola (Guinea 2014 fit)",
        model_hint="SEIR-friendly preset",
        beta=0.270,
        gamma=0.178,
        sigma=0.189,
        note="Directly based on the Guinea estimates reported by Althaus (2014): β = 0.27 day^-1, 1/σ = 5.3 days, and 1/γ = 5.61 days.",
        citations=(
            PresetCitation(
                label="Althaus 2014, PLOS Currents Outbreaks",
                url="https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/alertasActuales/ebola/docs/4.EstimatingtheReproductionNumberofEbola2014OutbreakinWestAfrica.pdf",
            ),
        ),
    ),
    DiseasePreset(
        key="measles_classic",
        disease="Measles (classic community spread)",
        model_hint="SEIR-to-SIR preset",
        beta=1.875,
        gamma=0.125,
        sigma=0.087,
        note="Derived for a simple SEIR model using an R0 midpoint of 15 from the 12-18 range in Guerra et al. (2017), a mean incubation period of roughly 11-12 days from Klinkenberg and Nishiura (2011), and an 8-day infectious window from 4 days before to 4 days after rash onset reported in Gastanaduy et al. (2023).",
        citations=(
            PresetCitation(
                label="Guerra et al. 2017, measles R0 systematic review",
                url="https://pubmed.ncbi.nlm.nih.gov/28757186/",
            ),
            PresetCitation(
                label="Klinkenberg and Nishiura 2011, incubation and generation time",
                url="https://pubmed.ncbi.nlm.nih.gov/21704640/",
            ),
            PresetCitation(
                label="Gastanaduy et al. 2023, infectious window used in outbreak response",
                url="https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(23)00130-5/fulltext",
            ),
        ),
    ),
)

PRESET_BY_KEY = {preset.key: preset for preset in DISEASE_PRESETS}


def preset_options() -> list[dict[str, str]]:
    return [
        {
            "label": f"{preset.disease} · β={preset.beta:.3f}, γ={preset.gamma:.3f}"
            + (f", σ={preset.sigma:.3f}" if preset.sigma is not None else ""),
            "value": preset.key,
        }
        for preset in DISEASE_PRESETS
    ]
