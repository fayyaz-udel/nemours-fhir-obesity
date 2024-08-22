const debug = true;
const inference_server = "http://localhost:4000"; // "https://wlmresfhr500.nemours.org/nemours-fhir-obesity/inference/";
const fhir_server = "http://localhost:3000"; // "https://wlmresfhr500.nemours.org/nemours-fhir-obesity/fhir/";
function communicate_server(data) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", inference_server);

        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onload = function () {
            if (xhr.status === 200) {
                resolve(xhr.responseText); //SUCCESS
            } else {
                reject(new Error(xhr.statusText)); //ERROR
            }
        };
        xhr.onerror = function () {
            reject(new Error("Network error"));
        };
        xhr.send(JSON.stringify(data));
    });
}


function get_demographic(pat_id) {
    if (debug) {
        const client = FHIR.client(fhir_server);
        return client.request("/Patient?_id=" + pat_id).then(function (pt) {
            pt = pt.entry[0].resource;
            return pt;
        });

    } else {

    }
    return FHIR.oauth2.ready().then(function (client) {
        return client.request("/Patient?_id=" + pat_id).then(function (pt) {
            pt = pt.entry[0].resource;
            return pt;
        });

    }).catch(function (error) {
        console.error(error);
        throw error;
    });
}


function get_meds(pat_id) {
    if (debug) {
        const client = FHIR.client(fhir_server);
        return client.request("/MedicationRequest?patient=" + pat_id, {
            resolveReferences: ["medicationReference"], graph: true,
        })
            .then(function (data) {
                return data.entry;
            })
            .then(function (meds) {
                if (!meds) {
                    return null;
                }
                if (!meds.length) {
                    return null;
                }
                let table = document.createElement('table')
                for (var j = 0; j < meds.length; j++) {
                    let row = table.insertRow();

                    let date = row.insertCell();
                    date.textContent = meds[j]["resource"]["authoredOn"];

                    let code = row.insertCell();
                    code.textContent = meds[j]["resource"]["medicationCodeableConcept"]["coding"][0]["code"];

                    let name = row.insertCell();
                    name.textContent = meds[j]["resource"]["medicationCodeableConcept"]["coding"][0]["display"];

                }
                document.getElementById("meds").insertAdjacentElement("afterend", table);

                return meds;
            });

    } else {
        return FHIR.oauth2.ready().then(function (client) {

            return client.request("/MedicationRequest?patient=" + pat_id, {
                resolveReferences: ["medicationReference"], graph: true,
            })
                .then(function (data) {
                    return data.entry;
                })
                .then(function (meds) {
                    if (!meds) {
                        return null;
                    }
                    if (!meds.length) {
                        return null;
                    }
                    let table = document.createElement('table')
                    for (var j = 0; j < meds.length; j++) {
                        let row = table.insertRow();

                        let date = row.insertCell();
                        date.textContent = meds[j]["resource"]["authoredOn"];

                        let code = row.insertCell();
                        code.textContent = meds[j]["resource"]["medicationCodeableConcept"]["coding"][0]["code"];

                        let name = row.insertCell();
                        name.textContent = meds[j]["resource"]["medicationCodeableConcept"]["coding"][0]["display"];

                    }
                    document.getElementById("meds").insertAdjacentElement("afterend", table);

                    return meds;
                });
        }).catch(function (error) {
            document.getElementById("meds").innerText = error.stack;
            throw error;
        });
    }

}

function get_conds(pat_id) {
    if (debug) {
        const client = FHIR.client(fhir_server);

        return client.request("/Condition?patient=" + pat_id, {
            resolveReferences: ["conditionReference"], graph: true,
        })
            .then(function (data) {
                return data.entry;
            })
            .then(function (conds) {
                if (!conds) {
                    return null;
                }
                if (!conds.length) {
                    return null;
                }
                let table = document.createElement('table');
                for (var j = 0; j < conds.length; j++) {
                    let row = table.insertRow();

                    let date = row.insertCell();
                    date.textContent = conds[j]["resource"]["onsetDateTime"];

                    let code = row.insertCell();
                    code.textContent = conds[j]["resource"]["code"]["coding"][0]["code"];

                    let name = row.insertCell();
                    name.textContent = conds[j]["resource"]["code"]["coding"][0]["display"];

                }
                document.getElementById("conds").insertAdjacentElement("afterend", table);

                return conds;
            });
    } else {
        return FHIR.oauth2.ready().then(function (client) {
            return client.request("/Condition?patient=" + pat_id, {
                resolveReferences: ["conditionReference"], graph: true,
            })
                .then(function (data) {
                    console.log(data);
                    return data.entry;
                })
                .then(function (conds) {
                    if (!conds) {
                        return null;
                    }
                    if (!conds.length) {
                        return null;
                    }
                    let table = document.createElement('table');
                    for (var j = 0; j < conds.length; j++) {
                        let row = table.insertRow();

                        let date = row.insertCell();
                        date.textContent = conds[j]["resource"]["onsetDateTime"];

                        let code = row.insertCell();
                        code.textContent = conds[j]["resource"]["code"]["coding"][0]["code"];

                        let name = row.insertCell();
                        name.textContent = conds[j]["resource"]["code"]["coding"][0]["display"];

                    }
                    document.getElementById("conds").insertAdjacentElement("afterend", table);

                    return conds;
                });
        }).catch(function (error) {
            document.getElementById("conds").innerText = error.stack;
            throw error;
        });
    }


}


function get_obsrvs(pat_id) {
    if (debug) {
        const client = FHIR.client(fhir_server);
        return client.request("/Observation?patient=" + pat_id, {
            resolveReferences: ["medicationReference"], graph: true,
        })
            .then(function (data) {
                return data.entry;
            })
            .then(function (obrvs) {
                if (!obrvs) {
                    return null;
                }
                if (!obrvs.length) {
                    return null;
                }
                let table = document.createElement('table');
                for (var j = 0; j < obrvs.length; j++) {
                    if (!obrvs[j]["resource"]["code"]["coding"][0]["display"].includes("Body")) {
                        continue;
                    }
                    let row = table.insertRow();

                    let date = row.insertCell();
                    date.textContent = obrvs[j]["resource"]["effectiveDateTime"];

                    let row_no = row.insertCell();
                    row_no.textContent = j;

                    let code = row.insertCell();
                    code.textContent = obrvs[j]["resource"]["code"]["coding"][0]["code"];

                    let name = row.insertCell();
                    name.textContent = obrvs[j]["resource"]["code"]["coding"][0]["display"];

                    let value = row.insertCell();
                    value.textContent = obrvs[j]["resource"]["valueQuantity"]["value"].toFixed(2);

                }
                document.getElementById("obsrvs").insertAdjacentElement("afterend", table);

                return obrvs;
            });

    } else {
        return FHIR.oauth2.ready().then(function (client) {

            return client.request("/Observation?patient=" + pat_id, {
                resolveReferences: ["medicationReference"], graph: true,
            })
                .then(function (data) {
                    return data.entry;
                })
                .then(function (obrvs) {
                    if (!obrvs) {
                        return null;
                    }
                    if (!obrvs.length) {
                        return null;
                    }
                    let table = document.createElement('table');
                    for (var j = 0; j < obrvs.length; j++) {
                        if (!obrvs[j]["resource"]["code"]["coding"][0]["display"].includes("Body")) {
                            continue;
                        }
                        let row = table.insertRow();

                        let date = row.insertCell();
                        date.textContent = obrvs[j]["resource"]["effectiveDateTime"];

                        let row_no = row.insertCell();
                        row_no.textContent = j;

                        let code = row.insertCell();
                        code.textContent = obrvs[j]["resource"]["code"]["coding"][0]["code"];

                        let name = row.insertCell();
                        name.textContent = obrvs[j]["resource"]["code"]["coding"][0]["display"];

                        let value = row.insertCell();
                        value.textContent = obrvs[j]["resource"]["valueQuantity"]["value"].toFixed(2);

                    }
                    document.getElementById("obsrvs").insertAdjacentElement("afterend", table);

                    return obrvs;
                });
        }).catch(function (error) {
            document.getElementById("obsrvs").innerText = error.stack;
            throw error;
        });
    }

}


function get_server_response(data) {
    data = JSON.parse(data);

    document.getElementById("result").innerHTML = data['result'];
    document.getElementById("risk").innerHTML = data['risk'];
    document.getElementById("name").innerHTML = data['name'];
    document.getElementById("dob").innerHTML = data['dob'];
    document.getElementById("pop3").innerHTML = data['pop3'];
    document.getElementById('plot').src = 'data:image/png;base64,' + data['img'];

}





