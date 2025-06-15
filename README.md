# Energy Consumption Analysis Dataset

## Overview
This dataset contains measurements of appliance energy consumption along with environmental conditions collected at 10-minute intervals. The data can be used for energy consumption analysis and prediction.

## Data Description

### Time Period
- Data collected in 10-minute intervals 
- Format: YYYY-MM-DD HH:MM:SS

### Features
1. **Energy Consumption**
   - Appliances energy consumption in Wh
   - Light energy consumption in Wh

2. **Temperature Readings**
   - T1 to T9: Temperature measurements from different sensors
   - T_out: Outdoor temperature

3. **Humidity Measurements**
   - RH_1 to RH_9: Relative humidity from different sensors 
   - RH_out: Outdoor relative humidity

4. **Weather Conditions**
   - Press_mm_hg: Pressure in mm Hg
   - Windspeed
   - Visibility 
   - Tdewpoint: Dew point temperature

5. **Additional Variables**
   - rv1, rv2: Random variables

## File Format
- CSV (Comma-Separated Values)
- Each row represents one time interval
- Missing values are marked accordingly

## Potential Uses
- Energy consumption prediction
- Environmental analysis
- Building energy efficiency studies
- Time series analysis

## Data Location
The dataset is located in the `data` folder of this project.