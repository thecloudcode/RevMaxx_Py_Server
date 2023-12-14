from dotenv import load_dotenv
import requests
import os
from time import sleep

load_dotenv()

API_URL_output = "https://api-inference.huggingface.co/models/medical-ner-proj/bert-medical-ner-proj"
headers = {"Authorization": os.getenv("KEY")}
# requests.post(API_URL_output, headers=headers_output, json="")

API_URL_flan_t5 = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
# requests.post(API_URL_flan_t5, headers=headers_flan_t5, json="")

API_URL_summary = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# requests.post(API_URL_summary, headers=headers_summary, json="")

# input_text = input()
# final = []

def query_summary(payload):
	response = requests.post(API_URL_summary, headers=headers, json=payload)
	return response.json()

def query_flan_t5(payload):
	response = requests.post(API_URL_flan_t5, headers=headers, json=payload)
	return response.json()


def query_output(payload):
	response = requests.post(API_URL_output, headers=headers, json=payload)
	return response.json()



# assessment = query_flan_t5({
# 	"inputs": f"Generate the plan that should be given for treatment of {objective}",
# })


# print(treatments)
# print("\nASSESSMENT & PLAN")
# print(objective_treatments[0]['generated_text']+" "+objective_scans[0]['generated_text']+" "+plan[0]['generated_text'])


# print(json.dumps(final))

def execute(input_text):
	final = []
	if len(input_text) <= 110:
		summary = query_summary({
			"inputs": input_text,
		})
		# print("\nSUBJECTIVE")
		# print(summary[0]['summary_text'].replace("\\", ""))
		x = ["SUBJECTIVE"]
		x.append(summary[0]['summary_text'].replace("\\", ""))
		final.append(x)

	else:
		dots = []
		for i in range(len(input_text)):
			if input_text[i] == '.':
				dots.append(i)
		#
		summary = []
		l = len(dots)
		summary.append(
			query_flan_t5({"inputs": f"Generate a small meaningful summary on: {input_text[0:dots[l // 2] + 1:]}", })[
				0]['generated_text'])
		sleep(1)
		summary.append(
			query_flan_t5({"inputs": f"Generate a small meaningful summary on: {input_text[dots[l // 2] + 2::]}", })[0][
				'generated_text'])
		# print("\nSUBJECTIVE")
		# print(' '.join(summary))
		x = ["SUBJECTIVE"]
		x.append(' '.join(summary))
		final.append(x)
	output = query_output({
		"inputs": input_text,
	})
	p = list(set([item['word'] for item in output if item['entity_group'] in ['B_problem', 'I_problem']]))
	problems = []
	for i in p:
		problems.append(i) if i[0] != '#' else problems.append(i[2::])
	treatments = list(set([item['word'] for item in output if item['entity_group'] in ['I_treatment', 'B_treatment']]))
	test = list(set([item['word'] for item in output if item['entity_group'] in ['I_test', 'B_test']]))

	objective = []
	objective.append(query_flan_t5({
		"inputs": f"Generate a meaningful statement stating that these problems are faced by the patient: {', '.join(problems[:len(problems) // 2:])}",
	})[0]['generated_text'])
	sleep(1)
	objective.append(query_flan_t5({
		"inputs": f"Generate a meaningful statement stating that these problems are faced by the patient: {', '.join(problems[len(problems) // 2::])}",
	})[0]['generated_text'])
	# print("\nOBJECTIVE")
	# objective = ' '.join(objective)
	# print(objective)
	x = ["OBJECTIVE"]
	x.append(' '.join(objective))
	final.append(x)

	if test == []:
		test = ['no scans are done']
	if treatments == []:
		treatments = ['no treatments']

	objective_treatments = query_flan_t5({
		"inputs": f"Generate a meaningful statement about treatments: {', '.join(treatments)}",
	})
	objective_scans = query_flan_t5({
		"inputs": f"Generate a meaningful statement about the tests: {', '.join(test)}",
	})
	plan = query_flan_t5({
		"inputs": f"Generate a meaningful Plan for: {objective}",
	})

	x = ["ASSESSMENT & PLAN"]
	x.append(objective_treatments[0]['generated_text'] + " " + objective_scans[0]['generated_text'] + " " + plan[0][
		'generated_text'])
	final.append(x)

	final_json = {i[0]: i[1] for i in final}
	# print(final)
	return final_json

def get_SOAP(input_text):
	return execute(input_text)

print(get_SOAP(input()))