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
import h5py
import numpy as np

def setup():
    """Check existence of files"""
    print("Checking the configuration file")
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

def first_pass_drug_file(drug_csv_file_name, path_drug_name_dict, path_prescriber_name_dict, rewrite):
    if not(os.path.exists(path_drug_name_dict)) or rewrite:
        drug_name_dict = {}
        prescriber_id_dict = {}
        maximum_claim_count = 0

        with open(drug_csv_file_name, "rb") as fc:
            dict_reader = csv.DictReader(fc)

            for row_dict in dict_reader:
                drug_name = row_dict["BN"]
                prescriber_id = row_dict["UNIQUE_PRSCRBR_ID"]
                claim_count = int(row_dict["CLAIM_COUNT"])

                if drug_name in drug_name_dict:
                    drug_name_dict[drug_name] += 1
                else:
                    drug_name_dict[drug_name] = 1

                if prescriber_id in prescriber_id_dict:
                    prescriber_id_dict[prescriber_id] += 1
                else:
                    prescriber_id_dict[prescriber_id] = 1

                maximum_claim_count = max(maximum_claim_count, claim_count)

            print("Number of distinct drugs %s, number of prescribers is %s and max count is %s" %
                  (len(drug_name_dict), len(prescriber_id_dict), maximum_claim_count))

            with open(path_drug_name_dict, "w") as fw:
                json.dump(drug_name_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))

            with open(path_prescriber_name_dict, "w") as fw:
                json.dump(prescriber_id_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))


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


def prescriber_file_to_dict(path_prescriber_file, path_prescriber_id_file, path_dict_prescriber_file, rewrite):
    """Creates an a dict"""
    if not(os.path.exists(path_dict_prescriber_file)) or rewrite:

        with open(path_prescriber_id_file, "r") as f:
            prescriber_id_dict = json.load(f)

        with open(path_prescriber_file, "rb") as fc:
            dict_reader = csv.DictReader(fc)

            prescriber_dict = {}
            for prescriber in dict_reader:
                prescriber_id = prescriber["UNIQUE_PRSCRBR_ID"]
                if prescriber_id in prescriber_id_dict:
                    keys_to_add = []
                    for field in prescriber:
                        if field != "UNIQUE_PRSCRBR_ID":
                            if len(prescriber[field]) > 0:
                                keys_to_add += [field]

                    prescriber_dict[prescriber_id] = {}
                    for key in keys_to_add:
                        prescriber_dict[prescriber_id][key] = prescriber[key]

            with open(path_dict_prescriber_file, "w") as fw:
                json.dump(prescriber_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))


def create_npi_in_file_dict(path_prescriber_dict_file, path_npi_dict_file, rewrite):
    """"""
    if not(os.path.exists(path_npi_dict_file)) or rewrite:

        with open(path_prescriber_dict_file, "r") as f:
            prescriber_dict = json.load(f)

        npi_dict = {}
        for prescriber_id in prescriber_dict:
            prescriber = prescriber_dict[prescriber_id]
            if "NPI_1" in prescriber:
                npi = prescriber["NPI_1"]
                if npi in npi_dict:
                    npi_dict[npi] += 1
                else:
                    npi_dict[npi] = 1

        with open(path_npi_dict_file, "w") as fw:
            json.dump(npi_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))


def extract_prescriber_details_from_nppes_abridged_file_name(nppes_file_name, path_npi_dict_file, abridged_file_name, name_map, abrdiged_field_order, rewrite):
    """Creates an abridges CSV file of the NPPES file for a subset of the file that has a matching NPI in the path_npi_dict_file"""

    if not(os.path.exists(abridged_file_name)) or rewrite:

        reverse_map = {}
        expanded_map = {}
        list_map = {}
        reverse_list_map = {}
        for field_name in name_map:

            mapping = name_map[field_name]
            if mapping.__class__ == [].__class__:
                base_field_name = mapping[0]
                start_i = mapping[1]
                end_i = mapping[2]

                mapping_list = []
                reverse_mapping_list = []
                for i in range(start_i, end_i + 1):
                    mapping_list += [base_field_name + str(i)]
                    reverse_mapping_list += [field_name + str(i)]

                list_map[field_name] = []
                reverse_list_map[base_field_name] = []

                for i in range(len(mapping_list)):
                    existing_field_name = reverse_mapping_list[i]
                    mapping_item = mapping_list[i]
                    reverse_map[mapping_item] = existing_field_name
                    expanded_map[existing_field_name] = mapping_item
                    list_map[field_name] += [mapping_item]
                    reverse_list_map[base_field_name] += [existing_field_name]
            else:
                expanded_map[field_name] = mapping
                reverse_map[mapping] = field_name

        with open(path_npi_dict_file, "r") as f:
            npi_dict_file = json.load(f)

        with open(nppes_file_name, "rb") as fc:
            nppes_dict_reader = csv.DictReader(fc)

            with open(abridged_file_name, "wb") as fwc:
                csv_writer = csv.writer(fwc)
                csv_writer.writerow(abrdiged_field_order)

                j = 0
                for nppes_dict in nppes_dict_reader:
                    npi_field_name = reverse_map["npi"]
                    npi = nppes_dict[npi_field_name]
                    if npi in npi_dict_file:
                        row_to_add = []
                        for abridged_field_name in abrdiged_field_order:
                            value = ""
                            if abridged_field_name in reverse_map:
                                field_name = reverse_map[abridged_field_name]

                                if field_name in nppes_dict:
                                    value = nppes_dict[field_name]
                            else:
                                if abridged_field_name in reverse_list_map:
                                    fields = reverse_list_map[abridged_field_name]
                                    value_list = []
                                    for field in fields:
                                        if field in nppes_dict:
                                            value = nppes_dict[field]
                                            if len(value):
                                                value_list += [value]
                                    value = ("|").join(value_list)

                            row_to_add += [value]
                        csv_writer.writerow(row_to_add)

                        if j % 100000 == 0:
                            print("Wrote %s lines" % j)
                        j += 1


def annotate_abridged_csv_file(nppes_abridged_csv_file, nppes_annotated_csv_file, rewrite):
    if not(os.path.exists(nppes_annotated_csv_file)) or rewrite:

        with open(nppes_abridged_csv_file, "rb") as fc:
            csv_reader = csv.reader(fc)
            header = csv_reader.next()

        with open(nppes_abridged_csv_file, "rb") as fc:
            csv_dict_reader = csv.DictReader(fc)

            header += ["cleaned_credentials", "primary_taxonomy", "zip5", "sort_key"]

            with open(nppes_annotated_csv_file, "wb") as fwc:
                csv_writer = csv.writer(fwc)
                csv_writer.writerow(header)
                j = 0
                for nppes_dict in csv_dict_reader:
                    row_to_add = []
                    for field_name in header:
                        if field_name in nppes_dict:
                            row_to_add += [nppes_dict[field_name]]

                    cleaned_credentials = "".join(nppes_dict["credentials"].split(".")).upper()
                    row_to_add += [cleaned_credentials]

                    taxonomy = nppes_dict["taxonomy"].split("|")
                    primary_taxonomy_indicator = nppes_dict["primary taxonomy indicator"].split("|")

                    if 'Y' in primary_taxonomy_indicator:
                        position = primary_taxonomy_indicator.index('Y')
                        primary_taxonomy = taxonomy[position]
                    else:
                        primary_taxonomy = taxonomy[0]

                    row_to_add += [primary_taxonomy]
                    zip_code = nppes_dict["zip"]
                    zip_code5 = zip_code[0:5]

                    row_to_add += [zip_code5]
                    state = nppes_dict["state"]
                    npi = nppes_dict["npi"]

                    sort_key = [state, primary_taxonomy, zip_code5, npi]
                    row_to_add += ["|".join(sort_key)]

                    csv_writer.writerow(row_to_add)

                    j += 1
                    if j % 100000 == 0:
                        print("Wrote %s lines" % j)


def generate_reverse_and_forward_npi_key_maps(nppes_annotated_csv_file, npi_key_forward_map_name,
                                              npi_key_reverse_map_name, rewrite):

    if not(os.path.exists(npi_key_forward_map_name)) or rewrite:

        npi_key_forward_dict = {}
        npi_key_reverse_dict = {}
        with open(nppes_annotated_csv_file, "rb") as fc:
            dict_reader = csv.DictReader(fc)

            for row in dict_reader:
                npi = row["npi"]
                sort_key = row["sort_key"]

                npi_key_forward_dict[npi] = sort_key
                npi_key_reverse_dict[sort_key] = npi

        with open(npi_key_forward_map_name, "w") as fw:
            json.dump(npi_key_forward_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))

        with open(npi_key_reverse_map_name, "w") as fw:
            json.dump(npi_key_reverse_dict, fw, sort_keys=True, indent=4, separators=(',', ': '))


def align_npi_prescriber_key(dict_prescriber_file, npi_key_forward_map_name, prescriber_id_to_position_name,
                             prescriber_id_to_sort_key_name, rewrite):
    """We want to create a prescriber id to position mapping"""

    if not(os.path.exists(prescriber_id_to_position_name)) or rewrite:

        with open(dict_prescriber_file, 'r') as f:
            prescriber_dict = json.load(f)

        prescriber_id_to_sort_key = {}
        prescriber_id_to_sort_key_reverse = {}

        with open(npi_key_forward_map_name, "r") as f:
            npi_key_forward_dict = json.load(f)

        for prescriber_id in prescriber_dict:
            prescriber = prescriber_dict[prescriber_id]
            if "NPI_1" in prescriber:
                npi = prescriber["NPI_1"]

                if npi in npi_key_forward_dict:
                    sort_key = npi_key_forward_dict[npi]
                else:
                    sort_key = "ZZ" + prescriber_id
                if sort_key[0:1] == "||":
                    sort_key = "ZZ" + prescriber_id
            else:
                sort_key = "ZZ" + prescriber_id

            prescriber_id_to_sort_key[prescriber_id] = sort_key
            prescriber_id_to_sort_key_reverse[sort_key] = prescriber_id


        sort_keys_list = prescriber_id_to_sort_key_reverse.keys()
        sort_keys_list.sort()

        prescriber_id_to_position = {}
        for i in range(len(sort_keys_list)):
            sort_key = sort_keys_list[i]
            prescriber_id = prescriber_id_to_sort_key_reverse[sort_key]

            prescriber_id_to_position[prescriber_id] = i

        with open(prescriber_id_to_position_name, "w") as fw:
            json.dump(prescriber_id_to_position, fw,  sort_keys=True, indent=4, separators=(',', ': '))

        with open(prescriber_id_to_sort_key_name, "w") as fw:
            json.dump(prescriber_id_to_sort_key, fw, sort_keys=True, indent=4, separators=(',', ': '))

def write_prescriber_array_to_hdf5(hdf5_file_name, drug_file_name, prescriber_position_dict_name,
                                   drug_name_position_dict_name,
                                   h5_path="medicare_drug/prescriber/claim_count", rewrite=False):

    if not(os.path.exists(hdf5_file_name)) or rewrite:
        with open(prescriber_position_dict_name, "r") as f:
            prescriber_position_dict = json.load(f)

        with open(drug_name_position_dict_name, "r") as f:
            drug_name_position_dict = json.load(f)

        number_of_prescribers = len(prescriber_position_dict.keys())
        number_of_drugs = len(drug_name_position_dict.keys())

        file5 = h5py.File(hdf5_file_name)
        compressed_data_set = file5.create_dataset(h5_path, shape=(number_of_prescribers, number_of_drugs),
                                                   dtype="uint16", compression="gzip")

        with open(drug_file_name, "rb") as f:
            drug_dict_reader = csv.DictReader(f)
            i = 0

            current_prescriber_id = None
            start_time = time.time()
            current_time = start_time
            for drug_dict in drug_dict_reader:
                claim_count = int(drug_dict["CLAIM_COUNT"])
                drug_name = drug_dict["BN"]
                prescriber_id = drug_dict["UNIQUE_PRSCRBR_ID"]

                prescriber_position = prescriber_position_dict[prescriber_id]
                drug_name_position = drug_name_position_dict[drug_name]
                compressed_data_set[prescriber_position, drug_name_position] = claim_count

                i += 1
                if i % 100000 == 0:
                    past_time = current_time
                    current_time = time.time()
                    print("Wrote %s entries in %s seconds" %(i, current_time - past_time))

            print("Wrote a total of %s lines in %s seconds" % (i-i, time.time() - start_time))


def main(rewrite=False):
    config_dict = setup()
    data_directory = config_dict["data_directory"]
    file_prefix = config_dict["file_prefix"]
    prescriber_file = file_prefix + "_" + "prescriber_details.csv"
    full_path_prescriber_file = os.path.join(data_directory, prescriber_file)

    prescriber_id_file = config_dict["files"]["path_to_prescriber_id_file"]
    convert_sasbdat7_to_csv(prescriber_id_file, full_path_prescriber_file, rewrite)

    part_d_file = config_dict["files"]["path_to_main_medicare_part_d_file"]
    main_drug_file = file_prefix + "_" + "main_drug_file.csv"
    full_path_main_drug_file = os.path.join(data_directory, main_drug_file)
    convert_sasbdat7_to_csv(part_d_file, full_path_main_drug_file, rewrite)

    path_drug_name_dict = os.path.join(data_directory, "drug_name_dict.json")
    path_prescriber_id_dict = os.path.join(data_directory, "prescriber_id_dict.json")
    first_pass_drug_file(full_path_main_drug_file, path_drug_name_dict, path_prescriber_id_dict, rewrite)

    path_drug_name_position_dict = os.path.join(data_directory, "drug_name_position_dict.json")
    write_drug_name_position_dict(path_drug_name_dict, path_drug_name_position_dict, rewrite)

    path_dict_prescriber_file = os.path.join(data_directory, "prescriber_dict.json")
    prescriber_file_to_dict(full_path_prescriber_file, path_prescriber_id_dict, path_dict_prescriber_file, rewrite)

    path_prescriber_npi_exists_dict_file = os.path.join(data_directory, "npi_exists_dict.json")
    create_npi_in_file_dict(path_dict_prescriber_file, path_prescriber_npi_exists_dict_file, rewrite)

    path_nppes_file_name = config_dict["files"]["path_to_nppes_file"]
    path_abridged_nppes_file_name = os.path.join(data_directory, "nppes_abridged.csv")

    nppes_mapping_dict = config_dict["nppes"]["field_name_mapping"]
    nppes_field_name_order = config_dict["nppes"]["field_name_order"]

    extract_prescriber_details_from_nppes_abridged_file_name(path_nppes_file_name, path_prescriber_npi_exists_dict_file,
                                                             path_abridged_nppes_file_name, nppes_mapping_dict,
                                                             nppes_field_name_order, rewrite)

    path_annotated_nppes_file_name = os.path.join(data_directory, "nppes_annotated.csv")
    annotate_abridged_csv_file(path_abridged_nppes_file_name, path_annotated_nppes_file_name, rewrite)

    path_npi_key_map_name = os.path.join(data_directory, "npi_map_key_dict.json")
    path_npi_key_reverse_map_name = os.path.join(data_directory, "npi_map_key_reverse_dict.json")

    generate_reverse_and_forward_npi_key_maps(path_annotated_nppes_file_name, path_npi_key_map_name,
                                              path_npi_key_reverse_map_name, rewrite)

    path_prescriber_id_to_position_name = os.path.join(data_directory, "prescriber_id_to_position.json")
    path_prescriber_id_to_sort_key_name = os.path.join(data_directory, "prescriber_id_to_sort_key.json")

    align_npi_prescriber_key(path_dict_prescriber_file, path_npi_key_map_name,
                             path_prescriber_id_to_position_name, path_prescriber_id_to_sort_key_name, rewrite)

    path_hdf5_file_name = os.path.join(data_directory, file_prefix + "_medicare_prescriber_matrix_gz.hdf5")
    write_prescriber_array_to_hdf5(path_hdf5_file_name, full_path_main_drug_file, path_prescriber_id_to_position_name,
                                   path_drug_name_position_dict,
                                   h5_path="medicare_drug/prescriber/claim_count")


if __name__ == "__main__":
    main()


"""
import h5py
DSMPD = h5py.File("partd2011_medicare_prescriber_matrix.hdf5")
DS = DSMPD["/medicare_drug/prescriber"]
DS.shape
DS = DSMPD["/medicare_drug/prescriber/"]
SMDP = DS[10000:11000:,:]
DS = DSMPD["/medicare_drug/prescriber/claim_count"]
DS.__class__
DS.shape
SMDP = DS[10000:11000:,:]
SMDP.sum(1)
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
plt.matshow(SMDP)
plt.show()
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric("jaccard")
PSMDPJ = dist.pairwise(SMDP)
plt.matshow(PSMDPJ)
plt.show()
ls *.json
import json
presc_keys_pos =  json.load(open("prescriber_id_to_sort_key.json")
)
presc_id_sort_keys =  json.load(open("prescriber_id_to_sort_key.json"))
presc_id_pos = json.load(open("prescriber_id_to_position.json")
)
presc_id_pos["3"]
presc_id_sort_keys["3"]
sort_keys = [presc_id_sort_keys[x] for x in presc_id_sort_keys]
len(sort_keys)
sort_keys.sort()
sort_keys_split = [x.split("|") for x in sort_keys]
sort_keys_split[0]
first_level_index_positions = {}
current_first_sort_key = sort_keys_split[0][0]
for sort_key in sort_keys_split:
    new_sort_key = sort_key[0]
    if new_sort_key != current_first_sort_key:
        first_level_index_positions[current_sort_key] = [start_i, i]
        start_i = i
        current_sort_key = new_sort_key
    i += 1
i = 0
current_sort_key = sort_keys_split[0][0]
for sort_key in sort_keys_split:
    if new_sort_key != current_sort_key:
        first_level_index_positions[current_sort_key] = [start_i, i]
for sort_key in sort_keys_split:
    new_sort_key =sort_key[0]
    if new_sort_key != current_sort_key:
        first_level_index_positions[current_sort_key] = [start_i, i-1]
        start_i = i
        current_sort_key = new_sort_key
    i+=1
i = 0
start_i = 0
for sort_key in sort_keys_split:
    new_sort_key =sort_key[0]
    if new_sort_key != current_sort_key:
        first_level_index_positions[current_sort_key] = [start_i, i-1]
        start_i = i
        current_sort_key = new_sort_key
    i += 1
first_level_index_positions["DE"]
x = range(0, 20)
x
x[3:10]
demp = first_level_index_positions["DE"]
demp
demp[1] -demp[0]
SMDP = DS[demp[0]:demp[1]+1,:]
SMPD.shape
SMDP.shape
plt.matshow(SMDP)
plt.show()
PSMDPJ = dist.pairwise(SMDP)
PSMDPJ = dist.pairwise(PSMDPJ)
PSMDPJ = dist.pairwise(SMDP)
plt.matshow(PSMDPJ)
plt.show()
sort_keys_split[demp[0]]
sort_keys_split[demp[0]-1]
sort_keys_split[demp[0]:demp[0]+50]
sort_keys_split[demp[0]:demp[0]+100]
SMDP[0:100].sum(0)
SMDP[0:100].sum(1)
SMDP.sum(1)
SMDP[0:100].sum(0).shape
plt.matshow(PSMDPJ)
plt.show()
sort_keys_split[demp[0]:demp[0]+1400]
sort_keys_split[demp[0]+1400]
set(0
)
sb.set()
sb.matrix?
sb.matrix?
sb.set_palette("Blues")
plt.matshow(PSMDPJ)
plt.show()
with sb.color_palette("Blues", 20):
    plt.matshow(PSMDPJ)
    plt.show()
sb.heatmap(PSMDPJ)
sb.show()
plot.show()
plt.show()
sb.set(style="dark")
plt.matshow(PSMDPJ)
plt.show()
plt.matshow(PSMDPJ, cmap="cubehelix")
plt.show()
cmap = sb.diverging_palette(220, 10, as_cmap=True)
plt.matshow(PSMDPJ, cmap=cmap)
plt.show()
import seaborn as sns
cmap =sns.light_palette("navy", reverse=True, as_cmap=True)
plt.matshow(PSMDPJ, cmap=cmap)
plt.show()
cmap =sns.dark_palette("purple", reverse=True, as_cmap=True)
plt.matshow(PSMDPJ, cmap=cmap)
plt.show()
cmap =sns.dark_palette("purple", reverse=False, as_cmap=True)
plt.matshow(PSMDPJ, cmap=cmap)
plt.show()
sns.palplot(sns.color_palette("BrBG", 100))
plt.show()
cmap = sns.color_palette("BrBG", 100, as_cmap =True)
cmap = sns.color_palette("BrBG", 100)
plt.matshow(PSMDPJ, cmap=cmap)
cmap = sns.diverging_palette(220, 20, n=20, as_cmap=True)
plt.matshow(PSMDPJ, cmap=cmap)
plt.show()
cmap =sns.light_palette("navy", reverse=True, as_cmap=True,n=50)
cmap =sns.light_palette("green", as_cmap=True)
plt.matshow(PSMDPJ, cmap=cmap)
plt.show()
cmap =sns.light_palette("green", as_cmap=True, reverse=True)
plt.matshow(PSMDPJ, cmap=cmap)
plt.show()
plt.hist(PSMDPJ[1000,:],bins=100)
plt.show()
plt.hist(PSMDPJ[500,:],bins=100)
plt.show()
plt.hist(PSMDPJ[:,:],bins=100)
plt.show()
demp
SMDP.shape
%hist
DS.shape
DS.len
DS.len()
NS.size
DS.size
DS.compression
demp
demp[1] - demp[0]
%hist
SMDP.sum()
SMDP > 1.sum()
np.sum(SMDP > 0)
import numpy as np
np.sum(SMDP > 0)
SMDP.shape
PSMDPJ.sum()
sort_keys_split[200]
sort_keys_split[1350]
sort_keys_split[1200]
sort_keys_split[1100]
sort_keys_split[1000]
sort_keys_split[demp[0]+1000]
sort_keys_split[demp[0]+400]
sort_keys_split[demp[0]+1400]
sort_keys_split[demp[0]+1350]
sort_keys_split[demp[0]+1300]
plt.line(PSMDPJ[200:])
plt.plot(PSMDPJ[200,:])
plt.show()
sort_keys_split[demp[0]+1350]
sort_keys_split[demp[0]+1350]
plt.plot(PSMDPJ[50,:])
plt.show()
plt.plot(1 - PSMDPJ[50,:])
plt.show()
plt.plot?
plt.plot?
plt.show()
plt.scatter(1 - PSMDPJ[50,:])
plt.scatter(np.arange(3510),1 - PSMDPJ[50,:])
plt.scatter(np.arange(3509),1 - PSMDPJ[50,:])
plt.scatter(np.arange(2170),1 - PSMDPJ[50,:])
plt.show()
plt.scatter(np.arange(2170),1 - PSMDPJ[200,:])
plt.show()
sort_keys_split[demp[0]+1318]
%hist
arp = first_level_index_positions["AR"]
arp
SMAP = DS[arp[0]:arp[1]+1,:]
SMAP.shape
JSMAPD = dist.pairwise(SMAP)
plt.matshow(JSMAPD)
plt.show()
sort_keys_split[demp[0]+1710]
sort_keys_split[demp[0]+1600]
sort_keys_split[demp[0]+1510]
sort_keys_split[demp[0]+1414]
sort_keys_split[demp[0]+2054]
sort_keys_split[demp[0]+2140]
sort_keys_split[demp[0]+230]
%hist
"""