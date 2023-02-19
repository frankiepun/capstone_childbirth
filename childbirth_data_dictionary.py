# data dictionary for the fields used in the model
import numpy as np

childbirth_data_dict = {}

childbirth_data_dict["birth_month"] = {
"1": "January",
"2": "February",
"3": "March",
"4": "April",
"5": "May",
"6": "June",
"7": "July",
"8": "August",
"9": "September",
"10": "October",
"11": "November",
"12": "December"
}

childbirth_data_dict["mother_age"] = {}
childbirth_data_dict["mother_age"]["12"]="10-12 years"
for i in range(13, 50):
    childbirth_data_dict["mother_age"][f"{i}"] = f"{i} years"
childbirth_data_dict["mother_age"]["50"]="50 years and over"

childbirth_data_dict["mother_nativity"] = {
"1": "Born in the U.S. (50 US States)",
"2": "Born outside the U.S. (includes possessions)",
"3": "Unknown or Not Stated"
}

childbirth_data_dict["residence_status"] = {
"1": "RESIDENT: State and county of occurrence and residence are the same",
"2": "INTRASTATE NONRESIDENT: State of occurrence and residence are the same but county is different",
"3": "INTERSTATE NONRESIDENT: State of occurrence and residence are different but both are one of the 50 US states or District of Columbia",
"4": "FOREIGN RESIDENT: The state of residence is not one of the 50 US states or District of Columbia."
}

childbirth_data_dict["mother_race1"] = {
"1": "White (only) [only one race reported] the United States except Puerto Rico",
"2": "Black (only)",
"3": "AIAN (American Indian or Alaskan Native) (only)",
"4": "Asian (only)",
"5": "NHOPI (Native Hawaiian or Other Pacific Islander) (only)",
"6": "Black and White",
"7": "Black and AIAN",
"8": "Black and Asian",
"9": "Black and NHOPI",
"10": "AIAN and White",
"11": "AIAN and Asian",
"12": "AIAN and NHOPI",
"13": "Asian and White",
"14": "Asian and NHOPI",
"15": "NHOPI and White",
"16": "Black, AIAN, and White",
"17": "Black, AIAN, and Asian",
"18": "Black, AIAN, and NHOPI",
"19": "Black, Asian, and White",
"20": "Black, Asian, and NHOPI",
"21": "Black, NHOPI, and White",
"22": "AIAN, Asian, and White",
"23": "AIAN, NHOPI, and White",
"24": "AIAN, Asian, and NHOPI",
"25": "Asian, NHOPI, and White",
"26": "Black, AIAN, Asian, and White",
"27": "Black, AIAN, Asian, and NHOPI",
"28": "Black, AIAN, NHOPI, and White",
"29": "Black, Asian, NHOPI, and White",
"30": "AIAN, Asian, NHOPI, and White",
"31": "Black, AIAN, Asian, NHOPI, and White"
}

childbirth_data_dict["mother_hispanic_race"] = {
"1": "Non-Hispanic White (only)",
"2": "Non-Hispanic Black (only)",
"3": "Non-Hispanic AIAN (only)",
"4": "Non-Hispanic Asian (only)",
"5": "Non-Hispanic NHOPI (only)",
"6": "Non-Hispanic more than one race",
"7": "Hispanic",
"8": "Origin unknown or not stated"
}

childbirth_data_dict["paternity_acknowledged"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown",
"X": "Not Applicable"
}

childbirth_data_dict["marital_status"] = {
"1": "Married",
"2": "Unmarried"
}

childbirth_data_dict["mother_education"] = {
"1": "8th grade or less",
"2": "9th through 12th grade with no diploma",
"3": "High school graduate or GED completed",
"4": "Some college credit, but not a degree.",
"5": "Associate degree (AA,AS)",
"6": "Bachelor’s degree (BA, AB, BS)",
"7": "Master’s degree (MA, MS, MEng, MEd, MSW, MBA)",
"8": "Doctorate (PhD, EdD) or Professional Degree (MD, DDS, DVM, LLB, JD)",
"9": "Unknown"
}

childbirth_data_dict["father_age"] = {}
childbirth_data_dict["father_age"]["12"]="10-12 years"
for i in range(13, 50):
    childbirth_data_dict["father_age"][f"{i}"] = f"{i} years"
childbirth_data_dict["father_age"]["50"]="50 years and over"
childbirth_data_dict["father_age"]["99"]="Unknown"

childbirth_data_dict["prior_births_now_living"] = {}
for i in range(0, 31):
    childbirth_data_dict["prior_births_now_living"][f"{i}"] = f"{i}"
childbirth_data_dict["prior_births_now_living"]["99"]="Unknown"

childbirth_data_dict["total_birth_order"] = {}
for i in range(1, 8):
    childbirth_data_dict["total_birth_order"][f"{i}"] = f"{i}"
childbirth_data_dict["total_birth_order"]["8"] = "8 or more total births"
childbirth_data_dict["total_birth_order"]["9"]="Unknown"

childbirth_data_dict["interval_since_last_live_birth"] = {
"0": "Zero to 3 months (plural delivery)",
"1": "4 to 11 months",
"2": "12 to 17 months",
"3": "18 to 23 months",
"4": "24 to 35 months",
"5": "36 to 47 months",
"6": "48 to 59 months",
"7": "60 to 71 months",
"8": "72 months and over",
"88": "Not applicable (1st live birth)",
"99": "Unknown or not stated"
}

childbirth_data_dict["month_prenatal_care_began"] = {
"1": "1st to 3rd month",
"2": "4th to 6th month",
"3": "7th to final month",
"4": "No prenatal care",
"5": "Unknown or not stated"
}

childbirth_data_dict["number_of_prenatal_visits"] = {
"1": "No visits",
"2": "1 to 2 visits",
"3": "3 to 4 visits",
"4": "5 to 6 visits",
"5": "7 to 8 visits",
"6": "9 to 10 visits",
"7": "11 to 12 visits",
"8": "13 to 14 visits",
"9": "15 to 16 visits",
"10": "17 to 18 visits",
"11": "19 or more visits",
"12": "Unknown or not stated"
}

childbirth_data_dict["wic"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["cigarettes_3rd_trimester"] = {}
for i in range(0, 99):
    childbirth_data_dict["cigarettes_3rd_trimester"][f"{i}"] = f"{i}"
childbirth_data_dict["cigarettes_3rd_trimester"]["99"] = "Unknown"

childbirth_data_dict["mother_height_in_total_inches"] = {}
for i in range(30, 79):
    childbirth_data_dict["mother_height_in_total_inches"][f"{i}"] = f"{i}"
childbirth_data_dict["mother_height_in_total_inches"]["99"] = "Unknown"

childbirth_data_dict["bmi"] = {}
for i in range(13, 70):
    childbirth_data_dict["bmi"][f"{i}"] = f"{i}"
childbirth_data_dict["bmi"]["99.9"] = "Unknown"

childbirth_data_dict["prepregnancy_weight"] = {}
for i in range(75, 376):
    childbirth_data_dict["prepregnancy_weight"][f"{i}"] = f"{i}"
childbirth_data_dict["prepregnancy_weight"]["999"] = "Unknown"

childbirth_data_dict["weight_gain_group"] = {
"1": "Less than 11 pounds",
"2": "11 to 20 pounds",
"3": "21 to 30 pounds",
"4": "31 to 40 pounds",
"5": "41 to 98 pounds",
"9": "Unknown"
}

childbirth_data_dict["gestational_diabetes"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["prepregnancy_hypertension"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["gestational_hypertension"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["previous_preterm_birth"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["infertility_treatment_used"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["fertility_enhancing_drugs"] = {
"Y": "Yes",
"N": "No",
"X": "Not applicable",
"U": "Unknown"
}

childbirth_data_dict["previous_cesarean"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["number_of_previous_cesareans"] = {}
for i in range(0, 31):
    childbirth_data_dict["number_of_previous_cesareans"][f"{i}"] = f"{i}"
childbirth_data_dict["number_of_previous_cesareans"]["99"]="Unknown"

childbirth_data_dict["no_risk_factors_reported"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["chlamydia"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

childbirth_data_dict["attendant_at_birth"] = {
"1": "Doctor of Medicine (MD)",
"2": "Doctor of Osteopathy (DO)",
"3": "Certified Nurse Midwife/Certified Midwife (CNM/CM)",
"4": "Other Midwife",
"5": "Other",
"9": "Unknown or not stated"
}

childbirth_data_dict["pluarality"] = {
"1": "Single",
"2": "Twin",
"3": "Triplet",
"4": "Quadruplet or higher"
}

childbirth_data_dict["sex_of_infant"] = {
"M": "Male",
"F": "Female"
}

childbirth_data_dict["last_normal_menses_month"] = {
"1": "January",
"2": "February",
"3": "March",
"4": "April",
"5": "May",
"6": "June",
"7": "July",
"8": "August",
"9": "September",
"10": "October",
"11": "November",
"12": "December",
"99": "Unknown"
}

childbirth_data_dict["infant_breastfed_at_discharge"] = {
"Y": "Yes",
"N": "No",
"U": "Unknown"
}

# it stores the user-friendly text used in HTML pages
childbirth_data_display_text_dict = {
    "birth_month" : "Expected childbirth month",
    "mother_age" : "Mother's age at childbirth",
    "mother_nativity" : "Mother's nativity",
    "residence_status" : "Mother's Residence status",
    "mother_race1" : "Mother's race",
    "mother_hispanic_race" : "Mother's hispanic origin",
    "paternity_acknowledged" : "Paternity acknowledged (father's name was identified)",
    "marital_status" : "Mother's marital status",
    "mother_education" : "Mother's education level",
    "father_age" : "Father's age",
    "prior_births_now_living" : "Number of children still living from previous live births",
    "total_birth_order" : "Expected birth order (for example, with two previous live births, enter 3)",
    "interval_since_last_live_birth" : "Interval since last live birth",
    "month_prenatal_care_began" : "The month prenatal care began",
    "number_of_prenatal_visits" : "The number of prenatal visits",
    "wic" : "Mother received special supplemental nutrition program for WIC (Women, Infants and Children)",
    "cigarettes_3rd_trimester" : "Cigarettes consumed in the 3rd trimester",
    "mother_height_in_total_inches" : "Mother's height in total inches",
    "bmi" : "Mother's BMI (Body Mass Index)",
    "prepregnancy_weight" : "Mother's prepregnancy weight",
    "weight_gain_group" : "Mother's expected weight gain at childbirth",
    "gestational_diabetes" : "Mother's gestational diabetes",
    "prepregnancy_hypertension" : "Mother's prepregnancy hypertension",
    "gestational_hypertension" : "Mother's gestational hypertension",
    "previous_preterm_birth" : "Mother's previous preterm birth",
    "infertility_treatment_used" : "Mother's infertility treatment used",
    "fertility_enhancing_drugs" : "Mother's fertility enhancing drugs",
    "previous_cesarean" : "Mother's previous cesarean",
    "number_of_previous_cesareans" : "Mother's number of previous cesareans",
    "no_risk_factors_reported" : "Mother has no other risk factors reported (Yes=No other risks factors, No=has other risks factor reported)",
    "chlamydia" : "Mother's chlamydia (a sexual transmitted disease)",
    "attendant_at_birth" : "Type of medical attendant at birth",
    "pluarality" : "Pluarality (Single, Twin, etc)",
    "sex_of_infant" : "Sex of infant",
    "last_normal_menses_month" : "Last normal menses month",
    "infant_breastfed_at_discharge" : "Infant breastfed at discharge"
}


# generaet sample HTMl snippets for field selection
def generate_html_input_form_select_field_sample_code():
    with open(f'C:/Users/frankiepun/Documents/workspace-w210-capstone/childbirth prediction/git/capstone_childbirth/DEBUG_html_form_select.txt', 'w') as f:
        for column_name in childbirth_data_dict:
            nested_dict = childbirth_data_dict[column_name]
            f.write(f'<label for="{column_name}">{childbirth_data_display_text_dict[column_name]}:</label>\n')
            f.write(f'<select id="{column_name}" name="{column_name}">\n')
            for key, value in nested_dict.items():
                f.write(f'  <option value="{key}" {{{{"selected" if default_value_dict.{column_name}=="{key}"}}}}>{value}</option>\n')
            f.write(f'</select>\n')
            f.write(f'<p>\n')

# generate_html_input_form_select_field_sample_code()
