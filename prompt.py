import argparse
import numpy as np
import torch
import jsonlines 
import random 

#---------------------------------------------------------------------------
def _inputToPrompt(model, instance, test=False, fewShot=False):
    if model=="idefics":
        prompt = []
        if test:
            if fewShot:
                prompt =[
                    "\nUser: Oh, I see! How about this image ",
                    instance["image"],
                    " with the caption '{caption}'?<end_of_utterance>".format(caption=instance["caption_choices"]),
                    "\nAssistant: ",
                ] 
            else:
                prompt =[
                    "User: I saw this image ",
                    instance["image"],
                    " in the New York Times with the caption '{caption}'".format(caption=instance["caption_choices"]),
                    "but I don't understand what the joke is! Can you explain it to me?<end_of_utterance>",
                    "\nAssistant: ", 
                ]
        else: 
            if fewShot:
                prompt =[
                    "User: I saw this image ",
                    instance["image"],
                    " in the New York Times with the caption '{caption}'".format(caption=instance["caption_choices"]),
                    "but I don't understand what the joke is! Can you explain it to me?<end_of_utterance>",
                    "\nAssistant: {explanation}<end_of_utterance>".format(explanation=instance["target"])
                ]
            else:
                prompt =[
                    "\nUser: Oh, I see! How about this image ",
                    instance["image"],
                    " with the caption '{caption}'?<end_of_utterance>".format(caption=instance["caption_choices"]),
                    "\nAssistant: {explanation}<end_of_utterance>".format(explanation=instance["target"]) 
                ] 
    elif model == "llama":
        #Example: instance["input"]
        #scene: the living room description: A man and a woman are sitting on a couch. They are surrounded by numerous monkeys. uncanny: Monkeys are found in jungles or zoos, not in houses. entities: Monkey, Amazon_rainforest, Amazon_(company). caption: Then maybe you should just tell me what you want for your birthday instead of saying you don't care.
        systemPrompt = """You are a helpful and a funny assistant."""
        sceneInd = instance["input"].index("scene:")
        descInd = instance["input"].index("description:")
        uncannyInd = instance["input"].index("uncanny:")
        entInd = instance["input"].index("entities:")
        capInd = instance["input"].index("caption:")
        scene = instance["input"][sceneInd+len("scene: "):descInd]
        description = instance["input"][descInd+len("description: "):uncannyInd]
        uncanny = instance["input"][uncannyInd+len("uncanny: "):entInd]
        caption = instance["input"][capInd+len("caption: "):]
        if test:
            if fewShot:
                prompt = "<s>Oh, I see! They also described me yet another scene in {scene}: {description} and said '{caption}'. Can you explain this one as well?[/INST]".format(
                    system_prompt=systemPrompt,
                    scene=scene,
                    description=description,
                    caption=caption,
                )
            else: 
                prompt = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Someone described me the following scene in {scene}: {description} and said '{caption}' but I didn't understand what the joke was. Can you explain it to me?[/INST]""".format(
                system_prompt=systemPrompt,
                scene=scene,
                description=description,
                caption=caption,
            )
        else:
            if fewShot:
                prompt = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Someone described me the following scene in {scene}: {description} and said '{caption}' but I didn't understand what the joke was. Can you explain it to me?[/INST] {model_answer} </s>""".format(
                    system_prompt=systemPrompt,
                    scene=scene,
                    description=description,
                    caption=caption,
                    model_answer=instance["target"],
                )
            else: 
                prompt = "<s>Oh, I see! They also described me another scene in {scene}: {description} and said '{caption}'. Can you explain this one as well?[/INST] {model_answer} </s>" .format(
                    system_prompt=systemPrompt,
                    scene=scene,
                    description=description,
                    caption=caption,
                    model_answer=instance["target"],
                )
    else: 
        raise ValueError("[_inputToPrompt] Unrecognized model: {}".format(model))
    return prompt
#---------------------------------------------------------------------------
def inputToPrompt(model, instance, fewShot=None):
    if model=="idefics" or model == "llama":
        prompt = []
        if fewShot:
            assert type(fewShot) == list
            for i, f in enumerate(fewShot):
                prompt.extend(_inputToPrompt(model, f, test=False, fewShot=(i==0)))
        
        prompt.extend(_inputToPrompt(model, instance, test=True, fewShot=fewShot))
        if model == "llama":
            prompt = "".join(prompt)
        return prompt
    else: 
        raise ValueError("[inputToPrompt] Unrecognized model: {}".format(model))
#---------------------------------------------------------------------------
def newyorker_caption_contest_data(args):
    from datasets import load_dataset
    dset = load_dataset(args.task_name, args.subtask)

    res = {}
    for spl, spl_name in zip([dset['train'], dset['validation'], dset['test']],
                            ['train', 'val', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['from_description']
            targ = inst['label']
            cur_spl.append({'input': inp, 'target': targ, 'instance_id': inst['instance_id'], 'image': inst['image'], 'caption_choices': inst['caption_choices']})
        
            #'input' is an image annotation we will use for a llama2 e.g. "scene: the living room description: A man and a woman are sitting on a couch. They are surrounded by numerous monkeys. uncanny: Monkeys are found in jungles or zoos, not in houses. entities: Monkey, Amazon_rainforest, Amazon_(company)."
            #'target': a human-written explanation 
            #'image': a PIL Image object
            #'caption_choices': is human-written explanation

        res[spl_name] = cur_spl
    return res
#---------------------------------------------------------------------------
def newyorker_caption_contest_idefics(args): 
    from transformers import IdeficsForVisionText2Text, AutoProcessor

    print("Loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IdeficsForVisionText2Text.from_pretrained(args.idefics_checkpoint, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(args.idefics_checkpoint)

    print("Loading data")
    nyc_data = newyorker_caption_contest_data(args)
    nyc_data_five_val = random.sample(nyc_data['val'],5)
    nyc_data_train_two = random.sample(nyc_data['train'],2)

    prompts = []

    for val_inst in nyc_data_five_val:
        # ======================> ADD YOUR CODE TO DEFINE A PROMPT WITH TWO TRAIN EXAMPLES/DEMONSTRATIONS/SHOTS <======================
        # Each instace has a key 'image' that contains the PIL Image. You will give that to the model as input to "show" it the image instead of an url to the image jpg file.
        
        prompts.append(inputToPrompt("idefics", val_inst, nyc_data_train_two))
        
        # I'm saving images to `out`` to be able to see them in the output folder
        val_inst['image'].save(f"out/{val_inst['instance_id']}.jpg")

    # --batched mode
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    # --single sample mode
    #inputs = processor(prompts[0], return_tensors="pt").to(device)

    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=1024)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    for i, t in enumerate(generated_text):
        print(f"{i}:\n{t}\n")
        gen_expl = t.split("Assistant:")[-1]
        nyc_data_five_val[i]['generated_idefics']=gen_expl

    # ======================> You will need to `mkdir out`
    filename = 'out/val.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_five_val:
            del item['image']
            writer.write(item)

    filename = 'out/train.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_train_two:
            del item['image']
            writer.write(item)
#---------------------------------------------------------------------------
def newyorker_caption_contest_llama2(args): 
    print ("Loading data")
    nyc_data_five_val = []
    with jsonlines.open('out/val.jsonl') as reader:
        for obj in reader:
            nyc_data_five_val.append(obj)

    nyc_data_train_two = []
    with jsonlines.open('out/train.jsonl') as reader:
        for obj in reader:
            nyc_data_train_two.append(obj)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("Loading model")
    '''
    Ideally, we'd do something similar to what we have been doing before: 

        tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.llama2_checkpoint, torch_dtype=torch.float16, device_map="auto")
        tokenizer.pad_token = tokenizer.unk_token_id
        
        prompts = [ "our prompt" for val_inst in nyc_data_five_val]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        output_sequences = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_text = [tokenizer.decode(s, skip_special_tokens=True) for s in output_sequences]

    But I cannot produce text with this prototypical code with HF llama2. 
    Thus we will use pipeline instead. 
    '''
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.llama2_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    for i, val_inst in enumerate(nyc_data_five_val):         
        # ======================> ADD YOUR CODE TO DEFINE A PROMPT WITH TWO TRAIN EXAMPLES/DEMONSTRATIONS/SHOTS <======================
        prompt = inputToPrompt("llama", val_inst, nyc_data_train_two)

        sequences = pipeline(
            prompt,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )
        
        gen_expl = sequences[0]['generated_text'].split("/INST] ")[-1]
        nyc_data_five_val[i]['generated_llama2']=gen_expl
        
        print(prompt)
        print(gen_expl)
        print("*"*20)

    filename = 'out/val.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_five_val:
            writer.write(item)
#---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random seed', default=14195422)
    parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
    parser.add_argument('--task_name', default="jmhessel/newyorker_caption_contest",  type=str, help='Name of the task that will be used by huggingface load dataset')    
    parser.add_argument('--subtask', default="explanation", type=str, help="The contest has three subtasks: matching, ranking, explanation")
    parser.add_argument('--idefics_checkpoint', default="HuggingFaceM4/idefics-9b-instruct", type=str, help="The hf name of an idefics checkpoint")
    parser.add_argument('--llama2_checkpoint', default="meta-llama/Llama-2-7b-chat-hf", type=str, help="The hf name of a llama2 checkpoint")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    newyorker_caption_contest_idefics(args)
    newyorker_caption_contest_llama2(args)