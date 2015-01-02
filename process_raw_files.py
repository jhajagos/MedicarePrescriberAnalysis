"""
    This takes the raw SAS files from:

    https://projects.propublica.org/data-store/
    https://projects.propublica.org/data-store/sets/health-mcd11-1

    And processes this in a format that will be used for more detailed analytics.
"""

import csv
import json
import os
import time
from sas7bdat import SAS7BDAT


def setup():
    """Check existence of files"""
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config_dict = json.load(f)

    else:
        with open("config.json.example", "r") as f:
            config_txt = f.read()

        with open("config.json", "w") as fw:
            fw.write(config_txt)

        with open("config.json", "r") as f:
            config_dict = json.load(f)

    for key in config_dict:

        if key == "files":
            for file_key in config_dict[key]:
                if os.path.exists(config_dict[key][file_key]):
                    print("Exists '%s'" % config_dict[key][file_key])
                else:
                    print("Does not exist '%s'" % config_dict[key][file_key])

        elif key == "data_directory":
            if os.path.exists(config_dict[key]):
                pass
            else:
                print("Making directory %s" % config_dict[key])
                os.makedirs(config_dict[key])

    return config_dict


def clean_sas_row(row):
    """SAS represents numbers with decimals so we convert ones that are integers to integers"""
    cleaned_row = []
    for element in row:
        if element.__class__ == float:
            if element - int(element) == 0:
                element = int(element)
        cleaned_row += [element]
    return cleaned_row


def convert_sasbdat7_to_csv(bdat7_filename, csv_filename, rewrite=False):
    """Converts a SAS BDAT7 file into a portable CSV. Also cleans up 0.0 and 4.0 to 0 and 4 respectively"""

    if rewrite or not(os.path.exists(csv_filename)):
        with SAS7BDAT(bdat7_filename) as f:
            with open(csv_filename, "wb") as fwc:
                csv_writer = csv.writer(fwc)

                i = 0
                start_time = time.time()
                last_time = start_time
                for row in f:
                    cleaned_row = clean_sas_row(row)
                    csv_writer.writerow(cleaned_row)

                    if i < 10:
                        print(cleaned_row)

                    if i % 100000 == 0 and i > 0:
                        current_time = time.time()
                        print("Read %s rows in %s seconds" % (i, current_time - last_time))
                        last_time = current_time

                    i += 1

                end_time = time.time()
                print("Read %s rows of '%s' in %s seconds and wrote as CSV to '%s'" % (i, bdat7_filename, end_time - start_time, csv_filename))

def first_pass_drug_file(drug_csv_file_name):

    drug_name_dict = {}
    maximum_claim_count = 0

    with open(drug_csv_file_name, "rb") as fc:
        dict_reader = csv.DictReader(fc)

        for row_dict in dict_reader:
            drug_name = row_dict["BN"]
            claim_count = int(row_dict["CLAIM_COUNT"])

            if drug_name in drug_name_dict:
                drug_name_dict[drug_name] += 1
            else:
                drug_name_dict[drug_name] = 1

            maximum_claim_count = max(maximum_claim_count, claim_count)


        print("Number of distinct drugs %s and max count is %s" % (len(drug_name_dict), maximum_claim_count))

        return drug_name_dict


def write_drug_name_dict(main_drug_filename, path_drug_name_dict, rewrite):
    if not(os.path.exists(path_drug_name_dict)) or rewrite:
        drug_name_dict = first_pass_drug_file(main_drug_filename)

        with open(path_drug_name_dict, "w") as fw:
            json.dump(drug_name_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))


def write_drug_name_position_dict(path_drug_name_dict, path_drug_name_position_dict, rewrite):
    if not(os.path.exists(path_drug_name_position_dict)) or rewrite:
        with open(path_drug_name_dict, "r") as f:
            drug_name_dict = json.load(f)

        drug_name_position_dict = {}

        drug_names = drug_name_dict.keys()
        drug_names.sort()

        for i in range(len(drug_names)):
            drug_name_position_dict[drug_names[i]] = i

        with open(path_drug_name_position_dict, "w") as fw:
             json.dump(drug_name_position_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))


def main(rewrite=False):
    config_dict = setup()
    data_directory = config_dict["data_directory"]
    file_prefix = config_dict["file_prefix"]
    prescriber_file = file_prefix + "_" + "prescriber_details.csv"
    full_path_precriber_file = os.path.join(data_directory, prescriber_file)

    prescriber_id_file = config_dict["files"]["path_to_prescriber_id_file"]
    convert_sasbdat7_to_csv(prescriber_id_file, full_path_precriber_file, rewrite)

    part_d_file = config_dict["files"]["path_to_main_medicare_part_d_file"]
    main_drug_file = file_prefix + "_" + "main_drug_file.csv"
    full_path_main_drug_file = os.path.join(data_directory, main_drug_file)
    convert_sasbdat7_to_csv(part_d_file, full_path_main_drug_file, rewrite)

    path_drug_name_dict = os.path.join(data_directory, "drug_name_dict.json")
    write_drug_name_dict(full_path_main_drug_file, path_drug_name_dict, rewrite)

    path_drug_name_position_dict = os.path.join(data_directory, "drug_name_position_dict.json")
    write_drug_name_position_dict(path_drug_name_dict, path_drug_name_position_dict, rewrite)




if __name__ == "__main__":
    main()



"""
from sas7bdat import SAS7BDAT
i = 0
with SAS7BDAT('pp_ccwid_bn_2011_r.sas7bdat') as f:
    for row in f:
        i += 1
        if i <= 10:
            print(row)
print(i)
cd ..
ls
cd propublica_prescriber_ids_2011/
ls
j = 0
with SAS7BDAT('propublica_prescriber_ids_2011.sas7bdat') as f:
    j += 1
    j = j -1
with SAS7BDAT('propublica_prescriber_ids_2011.sas7bdat') as f:
    j = 0
    for row in f:
        j += 1
        if j < 10:
            print(row)
print(j)
(j * j)
(j * j) * 1/2 - j
(j * j) * 1/2.0 - j
k = j - 1
10000 * 10000
((k * k) / 2.0) - k
k
k * 4
(k * k)
"""