
# <p align="center">Integratable Machine Learning for Obesity Risk Prediction using EHR Data</p>
  

    

## Abstract
Early prediction and intervention for obesity are pivotal in managing its associated comorbidities and improving patient outcomes. However, there is still a gap and discontinuity between obesity prediction tools and operational data systems in clinical settings. This project introduces a novel end-to-end system specifically designed for obesity prediction. Taking advantage of data routinely recorded in Electronic Health Records (EHRs), the model employs a diverse range of clinical, demographic, and medical variables to predict individual risk of obesity. A distinct feature of our system is its seamless integration capability with EHR systems using the Fast Healthcare Interoperability Resources (FHIR) standard. This ensures ease of adoption in varied healthcare settings, facilitating timely identification of at-risk patients and promoting early interventions.

        
##  Installation

```
cd ./web
docker image build -t web_image .
docker run -p 3000:3000 -d web_image

cd ./inference
docker image build -t engine_image .
docker run -p 4000:4000  -d engine_image


https://launch.smarthealthit.org

http://localhost:3000/launch.html
```   