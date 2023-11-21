from get_random_edge_sample import get_random_edge_sample
from openai import OpenAI

def ask_gpt(input_text, prompt, model, client):
    gpt_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model=model,
        temperature=0.0,
        
    )
    return gpt_response.choices[0].message.content


def evaluate_edge_helpfullness(edge, instruction, client):
    iPPI_helpfullness = None

    while iPPI_helpfullness is None:
        gpt_evaluation = ask_gpt(str(edge) + ":: Only give the number as the answer.", "From -100 to 100. 100 = very helpful, -100 = avoid. How helpful would inibiting the interaction between these uniprots in achiving this goal (make an educated guess and only respond with the number) Think about the whole interactome and all the effects it has in many steps: {}".format(instruction), "gpt-3.5-turbo", client)
        print(gpt_evaluation)
        try:
            iPPI_helpfullness = float(gpt_evaluation) / 100
        except:
            pass

    return iPPI_helpfullness 

def evaluate_edges(edge_list):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )

    instruction = ""
    with open("MULTISELS/instruction.txt", "r") as file:
        instruction = file.read()

    tuples = set()
    for edge in edge_list:
        iPPI_helpfullness = evaluate_edge_helpfullness(edge, instruction, client)
        tuples.add((edge[0], edge[1], iPPI_helpfullness))

    return tuples

def main():
    edges_to_evaluate = get_random_edge_sample(n=10)

    tuples = evaluate_edges(edges_to_evaluate)
    print(tuples)

if __name__ == "__main__":
    main()