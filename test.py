import json
import fhirclient.models.medicationrequest as p
with open('sample.json', 'r') as h:
    pjs = json.load(h)
patient = p.Patient(pjs)
patient.name[0].given
# prints patient's given name array in the first `name` property