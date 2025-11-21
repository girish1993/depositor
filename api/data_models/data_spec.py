from enum import Enum


class Job(Enum):
    admin = "admin."
    unknown = "unknown"
    unemployed = "unemployed"
    management = "management"
    housemaid = "housemaid"
    entrepreneur = "entrepreneur"
    student = "student"
    blue_collar = "blue-collar"
    self_employed = "self-employed"
    retired = "retired"
    technician = "technician"
    services = "services"


class Marital(Enum):
    married = "married"
    divorced = "divorced"
    single = "single"


class Education(Enum):
    unknown = "unknown"
    secondary = "secondary"
    primary = "primary"
    tertiary = "tertiary"


class YesNo(Enum):
    yes = "yes"
    no = "no"


class Month(Enum):
    jan = "jan"
    feb = "feb"
    mar = "mar"
    apr = "apr"
    may = "may"
    jun = "jun"
    jul = "jul"
    aug = "aug"
    sep = "sep"
    oct = "oct"
    nov = "nov"
    dec = "dec"


class Contact(Enum):
    unknown = "unknown"
    telephone = "telephone"
    cellular = "cellular"


class Poutcome(Enum):
    unknown = "unknown"
    other = "other"
    failure = "failure"
    success = "success"
