import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify
import io
import base64

def plot_bmi_percentiles(age_list,bmi_list, sex):
    age_range = [2,10]#[min(age_list)-1, max(age_list)+1]
    bmi_data = pd.read_csv("/var/www/nemours-fhir-obesity/inference/data/bmip.csv")
    # Filter data based on the provided age range and sex
    bmi_data['Age'] = bmi_data['Agemos'] / 12
    filtered_data = bmi_data[(bmi_data['Age'] >= age_range[0]) &
                             (bmi_data['Age'] <= age_range[1]) &
                             (bmi_data['Sex'] == sex)]

    if filtered_data.empty:
        print("No data available for the given age range and sex.")
        return

    # Plotting
    plt.figure(figsize=(12, 8))

    # Define the percentiles to plot
    percentiles = ['P5', 'P10', 'P25', 'P50', 'P75', 'P85', 'P90', 'P95', 'P97']
    plt.figure(facecolor='#FEF9EF')
    for percentile in percentiles:
        plt.plot(filtered_data['Age'], filtered_data[percentile], label=percentile)

        x_last = age_range[1]
        y_last = filtered_data[percentile][len(filtered_data[percentile])-1]
        plt.annotate(percentile[1:]+"%", xy=(x_last, y_last), xytext=(x_last+0.1, y_last-0.15),
                     fontsize=10, color='black', ha='left')

    plt.plot(age_list, bmi_list, 'o', color='#0D3F6B', markersize=10, alpha=0.75)

    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.ylim(12, 26)
    plt.xlim(age_range[0], age_range[1])
    plt.grid(True)
    #plt.savefig("/var/www/nemours-fhir-obesity/web/assets/chart.png", bbox_inches='tight')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    return image_base64

if __name__ == '__main__':
    al = [3, 4, 5, 6, 7, 8, 9]
    bmil = [14, 15, 16, 17, 18, 19, 20]
    plot_bmi_percentiles(al, bmil, sex=1)
