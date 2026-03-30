import csv
import os

def convert_tsv_to_csv(folder_path, tables):
    for filename in os.listdir(folder_path):
        # Controlla se il nome del file è nella lista delle tabelle
        if filename.split(".")[0] in tables and filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Leggi il file come TSV
            with open(file_path, "r", encoding="utf-8") as infile:
                sample = infile.read(2048)
                if "\t" not in sample:
                    continue  # Non è un TSV, salta

            print(f"Converting {filename}...")

            # Leggi e scrivi convertendo da tab a virgola
            # Create a new directory inside file_path
            new_dir_path = os.path.join(folder_path, "converted_files")
            os.makedirs(new_dir_path, exist_ok=True)
            temp_path = new_dir_path + ".tmp"
            with open(file_path, "r", encoding="utf-8") as infile, \
                 open(temp_path, "w", encoding="utf-8", newline='') as outfile:

                reader = csv.reader(infile, delimiter="\t")
                writer = csv.writer(outfile, delimiter=",")

                for row in reader:
                    writer.writerow(row)

            # Crea un nuovo file con il nome in minuscolo
            lower_case_filename = filename.lower()
            print(f"Renaming {filename} to {lower_case_filename}...")
            lower_case_file_path = os.path.join(new_dir_path, lower_case_filename)
            os.rename(temp_path, lower_case_file_path)

    print("Conversione completata.")

# Usa il percorso della tua directory OMOP qui
if __name__ == "__main__":
    folder = "/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/data/ehr/omop"
    tables = ["CONCEPT", "CONCEPT_ANCESTOR" "DRUG_STRENGTH", "CONCEPT_CLASS", "CONCEPT_SYNONYM", "CONCEPT_CPT4", "CONCEPT_RELATIONSHIP", "DOMAIN", "RELATIONSHIP", "VOCABULARY"]
    convert_tsv_to_csv(folder, tables)