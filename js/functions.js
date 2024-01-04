function communicate_server(data) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "http://localhost:4000");
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


function get_demographic() {
    return FHIR.oauth2.ready().then(function (client) {
        return client.patient.read().then(function (pt) {
            document.getElementById("p_first_name").innerHTML = pt.name[0].given;
            document.getElementById("p_last_name").innerHTML = pt.name[0].family;
            document.getElementById("p_dob").innerHTML = pt.birthDate;
            document.getElementById("p_gender").innerHTML = pt.gender;
            document.getElementById("p_ethnicity").innerHTML = pt.extension[1].extension[0].valueCoding.display;

            document.getElementById("p_race").innerHTML = pt.extension[0].extension[0].valueCoding.display;
            console.log("Hello, world!!!!");
            console.log(pt.extension[1].extension[0].valueCoding.display);
            return pt;
        });

    }).catch(function (error) {
        console.error(error);
        throw error;
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
        });
    }).catch(function (error) {
        document.getElementById("meds").innerText = error.stack;
        throw error;
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
        });
    }).catch(function (error) {
        document.getElementById("conds").innerText = error.stack;
        throw error;
    });
}


function get_obsrvs() {
    return FHIR.oauth2.ready().then(function (client) {
        return client.patient.read().then(function (pt) {
            const patientId = pt.id;

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
        });
    }).catch(function (error) {
        document.getElementById("obsrvs").innerText = error.stack;
        throw error;
    });
}


function draw_anthropometric_chart(data) {
    data = JSON.parse(data);

    document.getElementById("moc_data").innerHTML = data['moc_data'];


    let table = document.createElement('table')

    let row = table.insertRow();
    let time = row.insertCell();
    time.textContent = "Age (years)";
    let prob = row.insertCell();
    prob.textContent = "Probability of Obesity";

    for (var j = 0; j < data['time'].length; j++) {
        let row = table.insertRow();

        let time = row.insertCell();
        time.textContent = data['time'][j];

        let prob = row.insertCell();
        prob.textContent = data['prob'][j];

    }
    document.getElementById("preds").insertAdjacentElement("afterend", table);


    const bmiChartCtx = document.getElementById('bmiChart');
    new Chart(bmiChartCtx, {
        type: 'line',
        data: {
            labels: data['bmi_x'],
            datasets: [{
                label: 'BMI',
                data: data['bmi_y'],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Age (months)',
                        font:
                            {
                                size: 18  // Adjust the font size as needed
                            }
                    },
                    ticks: {
                        font: {
                            size: 18  // Adjust the tick font size as needed
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'BMI (kg/m2)',
                        font:
                            {
                                size: 18  // Adjust the font size as needed
                            }
                    },
                    ticks: {
                        font: {
                            size: 18  // Adjust the tick font size as needed
                        }
                    },
                    suggestedMax: 40,
                    suggestedMin: 10,
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: false  // Hides the dataset label
                }
            }
        }
    });


    // const heightChartCtx = document.getElementById('heightChart');
    // new Chart(heightChartCtx, {
    //     type: 'line',
    //     data: {
    //         labels: data['height_x'],
    //         datasets: [{
    //             label: 'Height',
    //             data: data['height_y'],
    //             borderWidth: 1
    //         }]
    //     },
    //     options: {
    //         scales: {
    //             y: {
    //                 beginAtZero: false
    //             }
    //         }
    //     }
    // });
    //
    // const weightChartCtx = document.getElementById('weightChart');
    // new Chart(weightChartCtx, {
    //     type: 'line',
    //     data: {
    //         labels: data['weight_x'],
    //         datasets: [{
    //             label: 'Weight',
    //             data: data['weight_y'],
    //             borderWidth: 1
    //         }]
    //     },
    //     options: {
    //         scales: {
    //             y: {
    //                 beginAtZero: false
    //             }
    //         }
    //     }
    // });

}





