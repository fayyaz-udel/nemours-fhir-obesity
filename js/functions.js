function get_demographic() {

    FHIR.oauth2.ready().then(function (client) {
        client.patient.read().then(function (pt) {
            document.getElementById("p_first_name").innerHTML = pt.name[0].given;
            document.getElementById("p_last_name").innerHTML = pt.name[0].family;
            document.getElementById("p_dob").innerHTML = pt.birthDate;
            document.getElementById("p_gender").innerHTML = pt.gender;
        });

    }).catch(console.error);
}

function get_meds() {
    FHIR.oauth2.ready().then(function (client) {
        client.patient.read().then(function (pt) {
            // Get MedicationRequests for the selected patient
            client.request("/MedicationRequest?patient=" + client.patient.id, {
                resolveReferences: ["medicationReference"],
                graph: true,
            })
                // Reject if no MedicationRequests are found
                .then(function (data) {
                    if (!data.entry || !data.entry.length) {
                        throw new Error("No medications found for the selected patient");
                    }

                    return data.entry;
                })

                // Render the current patient's medications (or any error)
                .then(
                    function (meds) {
                        document.getElementById("meds").innerText = JSON.stringify(meds, null, 4);
                    },
                    function (error) {
                        document.getElementById("meds").innerText = error.stack;
                    });
        })
    })
}

function get_conds() {
    FHIR.oauth2.ready().then(function (client) {
        client.patient.read().then(function (pt) {
            client.request("/Condition?patient=" + client.patient.id, {
                resolveReferences: ["conditionReference"],
                graph: true,
            })
                .then(function (data) {
                    if (!data.entry || !data.entry.length) {
                        throw new Error("No conditions found for the selected patient");
                    }

                    return data.entry;
                })

                .then(
                    function (conds) {
                        let table = document.createElement('table')
                        for (var j = 0; j < conds.length; j++) {
                            let row = table.insertRow();

                            let row_no = row.insertCell();
                            row_no.textContent = j;

                            let code = row.insertCell();
                            code.textContent = conds[j]["resource"]["code"]["coding"][0]["code"];

                            let name = row.insertCell();
                            name.textContent = conds[j]["resource"]["code"]["coding"][0]["display"];

                            let date = row.insertCell();
                            date.textContent = conds[j]["resource"]["onsetDateTime"];

                        }
                        document.getElementById("cond_head").insertAdjacentElement("afterend", table)

                    },
                    function (error) {
                        document.getElementById("conds").innerText = error.stack;
                    });

        })
    })
}


function get_obsrvs() {
    FHIR.oauth2.ready().then(function (client) {
        client.patient.read().then(function (pt) {


            // Get MedicationRequests for the selected patient
            client.request("/Observation?patient=" + client.patient.id, {
                resolveReferences: ["medicationReference"],
                graph: true,
            })
                // Reject if no MedicationRequests are found
                .then(function (data) {
                    if (!data.entry || !data.entry.length) {
                        throw new Error("No medications found for the selected patient");
                    }

                    return data.entry;
                })

                // Render the current patient's medications (or any error)
                .then(
                    function (obrvs) {







                        let table = document.createElement('table')
                        for (var j = 0; j < obrvs.length; j++) {
                            let row = table.insertRow();

                            let row_no = row.insertCell();
                            row_no.textContent = j;

                            let code = row.insertCell();
                            code.textContent = obrvs[j]["resource"]["code"]["coding"][0]["code"];

                            let name = row.insertCell();
                            name.textContent = obrvs[j]["resource"]["code"]["coding"][0]["display"];

                            // let date = row.insertCell();
                            // date.textContent = conds[j]["resource"]["onsetDateTime"];

                        }
                        document.getElementById("obsrv_head").insertAdjacentElement("afterend", table)






                        document.getElementById("obrvs").innerText = JSON.stringify(obrvs, null, 4);
                    },
                    function (error) {
                        document.getElementById("meds").innerText = error.stack;
                    });


        })

    })
}
