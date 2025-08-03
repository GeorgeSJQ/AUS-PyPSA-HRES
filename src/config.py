from enum import StrEnum

class REZIDS(StrEnum):
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    N4 = "N4"
    N5 = "N5"
    N6 = "N6"
    N7 = "N7"
    N8 = "N8"
    N9 = "N9"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    Q5 = "Q5"
    Q6 = "Q6"
    Q7 = "Q7"
    Q8 = "Q8"
    Q9 = "Q9"
    S1 = "S1"
    S2 = "S2"
    S3 = "S4"
    S5 = "S5"
    S6 = "S6"
    S7 = "S7"
    S8 = "S8"
    S9 = "S9"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"

class SOLARTYPE(StrEnum):
    PV = "SAT"
    CST = "CST"

class FREQUENCY(StrEnum):
    DAILY = "D"
    WEEKLY = "W"
    HOURLY = "h"
    HALFHOUR = "30min"

ACTIVETECH = ["PV", "WIND", "BESS", "GAS_RECIP", "BESS_1HR", "BESS_2HR", "BESS_4HR", "BESS_8HR", "BESS_12HR", "BESS_24HR", "OCGT_SML", "OCGT_LRG"]

tech_life = {
    "SOLAR_PV": 25,
    "WIND": 25,
    "BESS": 25,
    "GAS_RECIP": 25,
    "OCGT": 25,
    "OCGT_SML": 25,
    "OCGT_LRG": 25,
    "CCGT": 25,
    "CCGT_CCS": 25,
    "BIOMASS": 25,
}
class GeneratorSpecs:
    ramp_rates = {
        "OCGT_SML": 5220, #MW/hr
        "OCGT_LRG": 1320, #MW/hr
        "CCGT": 1320, #MW/hr
        "BIOMASS": 60, #MW/hr
        "GAS_RECIP": 2160, #MW/hr
        "CST": 360, #MW/hr
    }

    min_loads = {
        "OCGT_SML": 0.50, # 1/100
        "OCGT_LRG": 0.50, # 1/100
        "CCGT": 0.46, # 1/100
        "BIOMASS": 0.40, # 1/100
        "GAS_RECIP": 0, # 1/100
    }

    min_up_down_time = {
        "OCGT_SML": 1, # hours
        "OCGT_LRG": 1, # hours
        "CCGT": 4, # hours
        "BIOMASS": 0, # hours
        "GAS_RECIP": 0, # hours
    }


