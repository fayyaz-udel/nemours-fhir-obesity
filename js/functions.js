function get_preds(data) {

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:4000");
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify(data, null, 4));
    xhr.onload = function () {
        if (xhr.status === 200) {
            document.getElementById("preds").innerText = xhr.responseText;
        } else {
            console.log(xhr.statusText); //ERROR
        }
    };

}


function get_demographic() {
    return FHIR.oauth2.ready().then(function (client) {
        return client.patient.read().then(function (pt) {
            document.getElementById("p_first_name").innerHTML = pt.name[0].given;
            document.getElementById("p_last_name").innerHTML = pt.name[0].family;
            document.getElementById("p_dob").innerHTML = pt.birthDate;
            document.getElementById("p_gender").innerHTML = pt.gender;
            // Return the patient object to the next .then() in the chain
            return pt;
        });

    }).catch(function (error) {
        console.error(error);
        // Handle error or return a default value or promise rejection
        throw error; // Rethrow to keep the promise chain in a rejected state
    });
}


function get_meds() {
    return FHIR.oauth2.ready().then(function (client) {
        return client.patient.read().then(function (pt) {
            return client.request("/MedicationRequest?patient=" + client.patient.id, {
                resolveReferences: ["medicationReference"],
                graph: true,
            })
            .then(function (data) {
                if (!data.entry || !data.entry.length) {
                    throw new Error("No medications found for the selected patient");
                }
                return data.entry;
            })
            .then(function (meds) {
                // Create and return the table instead of just assigning to out_med
                let table = document.createElement('table')
                for (var j = 0; j < meds.length; j++) {
                    let row = table.insertRow();

                    let row_no = row.insertCell();
                    row_no.textContent = j;

                    let code = row.insertCell();
                    code.textContent = meds[j]["resource"]["medicationCodeableConcept"]["coding"][0]["code"];

                    let name = row.insertCell();
                    name.textContent = meds[j]["resource"]["medicationCodeableConcept"]["coding"][0]["display"];

                    let date = row.insertCell();
                    date.textContent = meds[j]["resource"]["authoredOn"];
                }
                document.getElementById("med_head").insertAdjacentElement("afterend", table);

                // Return the medication data for further processing if needed
                return meds;
            });
        });
    }).catch(function (error) {
        // Handle or display the error
        document.getElementById("meds").innerText = error.stack;
        throw error; // Re-throw the error in case we have further chaining
    });
}

function get_conds() {
    return FHIR.oauth2.ready().then(function (client) {
        return client.patient.read().then(function (pt) {
            const patientId = pt.id;  // Assuming 'id' is the correct property

            return client.request("/Condition?patient=" + patientId, {
                resolveReferences: ["conditionReference"],
                graph: true,
            })
            .then(function (data) {
                if (!data.entry || !data.entry.length) {
                    throw new Error("No conditions found for the selected patient");
                }
                return data.entry;
            })
            .then(function (conds) {
                let table = document.createElement('table');
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
                document.getElementById("cond_head").insertAdjacentElement("afterend", table);

                return conds;
            });
        });
    }).catch(function (error) {
        document.getElementById("conds").innerText = error.stack;
        throw error;
    });
}



function get_obsrvs() {
    return FHIR.oauth2.ready().then(function (client) {
        return client.patient.read().then(function (pt) {
            const patientId = pt.id; // Replace this with the correct way to get the patient's ID

            return client.request(`/Observation?patient=${patientId}`, {
                resolveReferences: ["medicationReference"],
                graph: true,
            })
            .then(function (data) {
                if (!data.entry || !data.entry.length) {
                    throw new Error("No observations found for the selected patient");
                }
                return data.entry;
            })
            .then(function (obrvs) {
                // It might be better to simply return the observations here
                // and do the DOM manipulation elsewhere.
                return obrvs.filter(obrv =>
                    obrv["resource"]["code"]["coding"][0]["display"].includes("Body"));
            });
        });
    }).catch(function (error) {
        document.getElementById("obsrvs").innerText = error.stack; // Make sure this ID matches your error display element
        throw error;
    });
}

// Example usage:
get_obsrvs().then(function (obrvs) {
    // Do the DOM manipulation here with the filtered observations
    console.log('Observations:', obrvs);
}).catch(function (error) {
    // Handle any errors here
    console.error('An error occurred:', error);
});


function draw_chart() {
    const ctx = document.getElementById('myChart');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
                label: '# of Votes',
                data: [12, 19, 3, 5, 2, 3],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

}





