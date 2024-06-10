import csv
import sys

def main(args):

  file_a, file_b, output_file = args

  with open(file_a) as csv_a:
    csv_reader_a = csv.DictReader(csv_a, delimiter=',')
    with open(file_b) as csv_b:
      csv_reader_b = csv.DictReader(csv_b, delimiter=',')
      with open(output_file, "w") as csv_outfile:
        fieldnames = ["id", "prompt", "response_a", "response_b"]
        csv_writer = csv.DictWriter(csv_outfile, fieldnames)
        csv_writer.writeheader()

        for row_a in csv_reader_a:
          row_b = next(csv_reader_b)
          if row_a["id"] != row_b["id"]:
            raise ValueError("Row id mismatch")
          row_out = {
              "id" : row_a["id"],
              "prompt" : row_a["prompt"],
              "response_a" : row_a["response"],
              "response_b" : row_b["response"],
          }
          csv_writer.writerow(row_out);


if __name__ == '__main__':
  main(sys.argv[1:])
