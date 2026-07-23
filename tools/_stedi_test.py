"""One-off Stedi 837P connectivity probe. Mock data, usageIndicator=T.
Delete after use. A rejection/validation error still proves the transport
+ auth reached Stedi (the claim was received and parsed)."""
import json
from app.compliance.adapters.stedi import StediAdapter

adapter = StediAdapter()
print("configured:", adapter.is_configured())
print("claims_url:", adapter.claims_url)

payload = {
    "controlNumber": "123456789",
    # Stedi Test Payer — test claims to it generate a 277CA and test ERAs,
    # and appear in the portal claims view with Test mode toggled ON.
    "tradingPartnerServiceId": "STEDI",
    "usageIndicator": "T",
    "submitter": {
        "organizationName": "Example Podiatry Group PLLC",
        "contactInformation": {"name": "Billing Office",
                               "phoneNumber": "5555550100"},
    },
    "receiver": {"organizationName": "Stedi Test Payer"},
    "subscriber": {
        "memberId": "MOCK123456789",
        "firstName": "Test",
        "lastName": "Patient",
        "dateOfBirth": "19800101",
        "gender": "M",
        "paymentResponsibilityLevelCode": "P",
    },
    "billing": {
        "providerType": "BillingProvider",
        "organizationName": "Example Podiatry Group PLLC",
        "npi": "1999999984",
        "employerId": "123456789",
        "taxonomyCode": "213E00000X",
        "address": {"address1": "123 Example Street", "city": "Exampleville",
                    "state": "FL", "postalCode": "33101"},
        "contactInformation": {"name": "Billing Office",
                               "phoneNumber": "5555550100"},
    },
    "claimInformation": {
        "claimFilingCode": "CI",
        "patientControlNumber": "TEST0001",
        "claimChargeAmount": "120.00",
        "placeOfServiceCode": "11",
        "claimFrequencyCode": "1",
        "signatureIndicator": "Y",
        "planParticipationCode": "A",
        "releaseInformationCode": "Y",
        "benefitsAssignmentCertificationIndicator": "Y",
        "healthCareCodeInformation": [
            {"diagnosisTypeCode": "ABK", "diagnosisCode": "M2151"}],
        "serviceLines": [{
            "serviceDate": "20260715",
            "professionalService": {
                "procedureIdentifier": "HC",
                "procedureCode": "99213",
                "lineItemChargeAmount": "120.00",
                "measurementUnit": "UN",
                "serviceUnitCount": "1",
                "compositeDiagnosisCodePointers": {
                    "diagnosisCodePointers": [1]},
            },
        }],
    },
}

res = adapter.submit_claim(payload)
print("\n=== SubmissionResult ===")
print("configured    :", res.configured)
print("submitted     :", res.submitted)
print("claim_reference:", res.claim_reference)
print("errors        :", json.dumps(res.errors, indent=2))
print("raw           :", json.dumps(res.raw, indent=2)[:2000]
      if res.raw else None)
