from typing import Annotated, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_models.data_spec import (
    Contact,
    Education,
    Job,
    Marital,
    Month,
    Poutcome,
    YesNo,
)


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # adding some bounds to the numerical attributes to ensure sanity in the incoming data points(based on educated assumptions)
    age: Annotated[int, Field(ge=18, le=110)]
    balance: Annotated[float, Field(ge=-1e9, le=1e9)]
    duration: Annotated[int, Field(ge=0, le=100000)]
    campaign: Annotated[int, Field(ge=0, le=1000)]
    pdays: Annotated[int, Field(ge=-1, le=5000)]
    previous: Annotated[int, Field(ge=0, le=10000)]
    day: Annotated[int, Field(ge=1, le=31)]

    # categorical attributes with their enum types
    job: Job
    marital: Marital
    education: Education
    default: YesNo
    housing: YesNo
    loan: YesNo
    contact: Contact
    month: Month
    poutcome: Poutcome

    @field_validator(
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
        mode="before",
    )
    @classmethod
    def normalise_case(cls, v):
        if isinstance(v, str):
            # make sure all string values are lowercased before validation
            return v.strip().lower()
        return v


class ApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: Annotated[List[Customer], Field(..., min_length=1)]


class ApiResponse(BaseModel):
    probabilities: List[float]
    predictions: List[int]
    labels: List[str]
