
import json
import os
import sys
# from sklearn.metrics import accuracy_score


def find_json_files(directory_path, name=""):
    """Recursively find all JSON files in the given directory and its subdirectories."""
    json_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
          if name=='ref_dev.json':
            if file == 'ref_dev.json':
                json_files.append(os.path.join(root, file))
          else:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    return json_files

def load_json(directory_path,  name=""):
    json_files = find_json_files(directory_path, name)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in the directory: {directory_path}")
    # Assuming we only need the first JSON file for this example
  
    with open(json_files[0], 'r', encoding='utf-8', errors='ignore') as file:
        return json.load(file)


def check_format(submissions, references):

    
    if len(submissions) != 2 * len(references):
        raise ValueError(f"Error: Submission size must be double the reference size. Found {len(submissions)} submissions and {len(references)} references.")
        # sys.exit(1)



    missing_context_ids = [reference['context_id'] for reference in references if reference['context_id'] not in [submission['context_id'] for submission in submissions]]
    if len(missing_context_ids) > 0:
        print(f"There are {len(missing_context_ids)} context ids missing:\n",missing_context_ids) 
        raise ValueError("There are missing context IDs.")
        # sys.exit(1)


def select_highest_ranking_submissions(submissions):
    highest_ranking_submissions = {}
    for submission in submissions:
        context_id = submission['context_id']
        if context_id not in highest_ranking_submissions or highest_ranking_submissions[context_id]['ranking_score'] < submission['ranking_score']:
            highest_ranking_submissions[context_id] = submission
    return list(highest_ranking_submissions.values())



def calculate_accuracy(submissions, references):
   
    correct = 0
    total = 0


    submissions = select_highest_ranking_submissions(submissions)
    sorted_references = sorted(references, key=lambda x: x['context_id'])
    sorted_submissions= sorted(submissions, key=lambda x: x['context_id'])

    # Iterate over sorted_submissions
    for reference, submission in zip(sorted_references, sorted_submissions):
        total += 1
        if reference['context_id'] == submission['context_id']:
            if reference['gloss_id'] == submission['gloss_id']:
                correct += 1
        else:
            raise ValueError("Mismatched context IDs in sorted_references and sorted_submissions.")

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    return accuracy


    # accuracy = accuracy_score(sorted_references['gloss_id'], sorted_submissions['gloss_id'])


def calculate_mrr_at_2(submissions, references):

    mrr_sum = 0

    for reference in references:
        matching_submissions = [s for s in submissions if s['context_id'] == reference['context_id']]
        matching_submissions.sort(key=lambda x: x['ranking_score'], reverse=True)

        ranks = [i + 1 for i, s in enumerate(matching_submissions[:2]) if s['gloss_id'] == reference['gloss_id']]

        if ranks:
            mrr_sum += 1 / ranks[0]

    mrr_at_2 = mrr_sum / len(references) if references else 0
    return mrr_at_2


def main(submission_file_path, reference_file_path, output_file_path):
#def main(submission_file_path, reference_file_path):
    submissions = load_json(submission_file_path)
    print("len submissions", len(submissions))
    references = load_json(reference_file_path, "ref_dev.json")
    print("len references", len(references))



        
    check_format(submissions, references)
    accuracy = calculate_accuracy(submissions, references)
    mrr_at_2 = calculate_mrr_at_2(submissions, references)

    # print(f'accuracy:{accuracy}')
    # print(f'mrr_at_2:{mrr_at_2}')

    output_file_path =open(os.path.join(output_file_path, 'scores.txt'),"w")

 
    output_file_path.write(f'accuracy: {accuracy:.4f}\n')
    output_file_path.write(f'mrr@2: {mrr_at_2:.4f}') 


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError("Usage: python scoring_program.py <submission_file_path> <reference_file_path> <output_file_path>")
        # sys.exit(1)

    submission_file_path = sys.argv[1]
    reference_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    main(submission_file_path, reference_file_path, output_file_path)
