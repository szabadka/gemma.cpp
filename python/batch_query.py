import csv
import sys
import subprocess

def main(args):

  input_file, col_name, output_file = args[:3]
  gemma_binary, model, weights, tokenizer = args[3:]

  with open(input_file) as csv_infile:
    csv_reader = csv.DictReader(csv_infile, delimiter=',')
    with open(output_file, "w") as csv_outfile:
      fieldnames = ["id", "prompt", "response"]
      csv_writer = csv.DictWriter(csv_outfile, fieldnames)
      csv_writer.writeheader()
      line_count = 0
      for row in csv_reader:
        p = subprocess.Popen([
            gemma_binary,
            "--model", model,
            "--weights", weights,
            "--tokenizer", tokenizer,
            "--verbosity", "0", "--num_threads", "48",
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        response = p.communicate(input=row[col_name].encode('utf-8'))[0]
        out = {
            "id" : "PROMPT%04d" % line_count,
            "prompt" : row[col_name],
            "response": response.decode(),
        }
        csv_writer.writerow(out);
        line_count += 1
  print(f'Processed {line_count} lines.')


if __name__ == '__main__':
  main(sys.argv[1:])
