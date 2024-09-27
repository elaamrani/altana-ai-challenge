import __mp_main__
from fastapi import FastAPI
from app.artifacts.country_classifier_classes_1 import (
    AddressCountryClassifier,
    TopNClassTransformer,
)
from joblib import load
from pydantic import BaseModel
from arango import ArangoClient

app = FastAPI()
client = ArangoClient(hosts="https://2ae4f052d710.arangodb.cloud:8529")
db = client.db(
    "machine_learning", username="lab_test", password="z-rRdN-Enf4qQwybGiVdbG"
)


@app.on_event("startup")
def load_model():
    setattr(__mp_main__, "TopNClassTransformer", TopNClassTransformer)
    setattr(__mp_main__, "AddressCountryClassifier", AddressCountryClassifier)
    app.country_model = load("/code/app/artifacts/country_classifier_pipeline_1.joblib")
    app.link_model = load("/code/app/artifacts/link_prediction_pipeline_1.joblib")


class AddressList(BaseModel):
    addresses: list[str]


class SiteIDs(BaseModel):
    from_site_id: str
    to_site_id: str


@app.post("/countries")
async def get_countries(address_list: AddressList, response_model=list):
    return {"countries": app.country_model.predict(address_list.addresses)}


@app.post("/links")
async def get_site_link_scores(site_ids: SiteIDs, response_model=list):
    features = [
        "min_country_line_rank",
        "max_country_line_rank",
        "min_country_page_rank",
        "max_country_page_rank",
        "min_country_effective_closeness",
        "max_country_effective_closeness",
        "min_site_line_rank",
        "max_site_line_rank",
        "min_site_page_rank",
        "max_site_page_rank",
        "min_site_effective_closeness",
        "max_site_effective_closeness",
        "min_organization_line_rank",
        "max_organization_line_rank",
        "min_organization_page_rank",
        "max_organization_page_rank",
        "min_organization_effective_closeness",
        "max_organization_effective_closeness",
    ]
    site_id_1 = site_ids.from_site_id
    site_id_2 = site_ids.to_site_id

    query = """
        LET siteIds = [@site_id_1, @site_id_2]

        FOR site IN sites
            FILTER site.site_id IN siteIds
            LET countryData = (
                FOR country IN countries
                    FILTER country._id == site.country_id
                    RETURN {
                        country_line_rank: country.line_rank,
                        country_page_rank: country.page_rank,
                        country_effective_closeness: country.effective_closeness
                    }
            )
            
            LET organizationData = (
                FOR org IN organizations
                    FILTER org._id == site.organization_id
                    RETURN {
                        organization_line_rank: org.line_rank,
                        organization_page_rank: org.page_rank,
                        organization_effective_closeness: org.effective_closeness,
                    }
            )
            
            RETURN {
                min_site_line_rank: MIN(site.line_rank),
                max_site_line_rank: MAX(site.line_rank),
                min_site_page_rank: MIN(site.page_rank),
                max_site_page_rank: MAX(site.page_rank),
                min_site_effective_closeness: MIN(site.effective_closeness),
                max_site_effective_closeness: MAX(site.effective_closeness),
                country: FIRST(countryData),
                organization: FIRST(organizationData)
            }
    """

    results = db.aql.execute(
        query, bind_vars={"site_id_1": site_id_1, "site_id_2": site_id_2}
    )
    # print(results.batch())

    return {"links": app.link_model.predict(list(results.batch()))}
