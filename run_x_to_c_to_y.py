import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import argparse
import os
import gc
import torch
from itertools import zip_longest, chain
import random as rnd
#from src.models.MONET import MONET 
#from src.models.CLIP import CLIPViTB16
#from src.models.BiomedCLIP import BiomedCLIP
from src.models.MMed_Llama_3_8B import MMedLlama3
from src.models.Explicd import Explicd
from src.models.Mistral import Mistral
from src.models.MedGemma import MedGemma
from src.models.GPT5 import GPT5
from src.models.Gemini import Gemini
from src.models.Qwen import Qwen
from src.utils import map_label_to_name, load_data, generate_template, convert_numbers_to_concepts, map_letter_to_label, calculate_metrics, save_data_to_json, seed_everything, get_current_date
from src.rices import RICES
from mmed_refiner import MMedBasedRefiner

# ==============================
# LOCAL CHECKPOINT PATH
# ==============================
MMED_CKPT_PATH = '/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/checkpoint/MMed-Llama-3-8B-EnIns'



clinical_concepts = [
            'typical pigment network',
            'atypical pigment network',
            'irregular streaks',
            'regular streaks',
            'regular dots and globules',
            'irregular dots and globules',
            'blue-whitish veil',
            'regression structures'
        ]


concept_reference_dict_MONET = {
    "Asymmetry": ["Symmetry", "Regular", "Uniform"],
    "Irregular": ["Regular", "Smooth"],
    "Black": ["White", "Creamy", "Colorless", "Unpigmented"],
    "Blue": ["Green", "Red"],
    "White": ["Black", "Colored", "Pigmented"],
    "Brown": ["Pale", "White"],
    "Erosion":["Deposition", "Buildup"],
    "Multiple Colors": ["Single Color", "Unicolor"],
    "Tiny": ["Large", "Big"],
    "Regular": ["Irregular"], 
}

concept_reference_dict_PH2 = {
    "typical pigment network": ["atypical pigment network", "absence of pigment network"],
    "atypical pigment network": ["typical pigment network", "regular pigment network"],
    "irregular streaks": ["regular streaks"],
    "regular streaks": ["irregular streaks"],
    "regular dots and globules": ["irregular dots and globules"],
    "irregular dots and globules": ["regular dots and globules"],
    "blue-whitish veil": ["clear", "non-whitish skin"],
    "regression structures": ["progression structures"]
}

concept_reference_dict_HAM10000 = {
    "thick reticular or branched lines": ["thin", "straight lines"],
    "black dots or globules in the periphery of the lesion": ["Absence of dots or globules in the center of the lesion"],
    "white lines or white structureless area": ["Dark lines or dark structured areas"],
    "eccentrically located structureless area": ["Centrally located structured area"],
    "grey patterns": ["Bright or colorful patterns"],
    "polymorphous vessels": ["Monomorphic vessels"],
    "pseudopods or radial lines at the lesion margin that do not occupy the entire lesional circumference": ["Smooth or uniform margin with a complete lesional circumference"],
    "asymmetric combination of multiple patterns or colours in the absence of other melanoma criteria": ["Symmetric combination of a single pattern or color with the presence of other melanoma criteria"],
    "melanoma simulator": ["Benign skin condition analyzer"],
}


def normalize_concepts_for_mmed(concepts_str: str, model_name: str) -> str:
    if model_name != "MMed":
        return concepts_str
    if "Thus the diagnosis is" in concepts_str:
        concepts_str = concepts_str[:concepts_str.find("Thus the diagnosis is")-1]
    return concepts_str.strip().rstrip('.')


def count_violations(report_str: str) -> int:
    from src.self_refiner.concept_refiner import ConceptConsistencyRules, ConceptSelfRefine
    rules = ConceptConsistencyRules()
    parser = ConceptSelfRefine(llm_refine_fn=None)   # only used for parse_concepts
    concepts_dict = parser.parse_concepts(report_str)
    return len(rules.check_consistency(concepts_dict))

def x_to_c(model_name: str, dataset:str, ckpt:str=None, split=None, raw_values=False, 
           concept_extractor:str=None, report_path: str = None, use_demos=False, 
           n_demos=0, ground_truth_concepts=False, refiner_name:str='mmed', 
           predict_for_train_set=False, data_path='data', concept_reference_dict: str = 'PH2',
           margin_threshold: float = 0.2):
    """Predicts concepts from MONET.

    Args:
        model_name (str): Name of the model. 
        dataset (str): Name of the dataset.
        concept_reference_dict (dict): Dictionary containing clinical concepts.
        split (int, optional): Split to use in the case of PH2. Defaults to None.
        raw_values (bool, optional): Whether to use raw concepts or not. Defaults to False.
        predict_for_train_set (bool, optional): Whether to generate the reports for training set.

    Returns:
        None: Save predicted concepts into a CSV file.
    """
    # ── refinement statistics collector ──────────────────────────────────────
    dict_refinement_stats = dict()
        # Set concept reference dictionary
    if concept_reference_dict == "PH2":          # Note: this variable comes from parameter
        concept_reference_dict = concept_reference_dict_PH2
    elif concept_reference_dict == "HAM10000":
        concept_reference_dict = concept_reference_dict_HAM10000
    else:
        concept_reference_dict = concept_reference_dict_HAM10000  # default

    if model_name == "MONET":
        concept_reference_dict = concept_reference_dict_MONET

    # Load data
    train_dataloader, test_dataloader = load_data(dataset=dataset, split=split, data_path=data_path)
    config = None
    mmed_refiner = None       
    # Initialize model
    if model_name == "MONET":
        model = MONET()
    elif model_name == "CLIP":
        model = CLIPViTB16()
    elif model_name == "BiomedCLIP":
        model = BiomedCLIP()
    
    elif model_name == "Explicd":
        from src.utils import create_explicd_config
        config = create_explicd_config(gpu_id=0)    # TODO: Make this dynamically
        model = Explicd(config=config)
        if refiner_name == 'mmed':
            from mmed_refiner import MMedBasedRefiner
            print("\n[INFO] Loading MMed refiner...")
            mmed_refiner = MMedBasedRefiner(ckpt=MMED_CKPT_PATH)
            print("[INFO] MMed refiner ready!")
        elif refiner_name == 'mistral':
            from mistral_refiner import MistralBasedRefiner
            print("\n[INFO] Loading Mistral refiner...")
            mmed_refiner = MistralBasedRefiner()
            print("[INFO] Mistral refiner ready!")
        elif refiner_name == 'rule':
            from src.self_refiner.concept_refiner import SimpleRuleBasedRefiner
            print("\n[INFO] Using rule-based refiner (no LLM).")
            mmed_refiner = SimpleRuleBasedRefiner()
        elif refiner_name == 'none':
            print("\n[INFO] No refinement.")
            mmed_refiner = None   
    else:
        raise TypeError(f"The specififed model {model_name} does not have a valid implementation.")
      
    # Get concept prompts
    if model_name == "MONET":
        prompt_info = {}
        for concept in concept_reference_dict.keys():
            prompt_info[concept] = model.get_prompt_embedding(concept_term_list=[concept])
            for ref_concept in concept_reference_dict[concept]:
                prompt_info[ref_concept] = model.get_prompt_embedding(concept_term_list=[ref_concept])

    dict_to_save_data = dict()
    for batch in tqdm(test_dataloader):
        img_ids = batch["img_id"]
        y_true = batch["class_label"].numpy()
        imgs = [Image.open(x) for x in batch["img_path"]]

        if model_name != "Explicd":
            image_embedding = model.extract_image_features(imgs)
            #image_features_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
            image_features_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)

        if model_name == "MONET":
            scores = model.get_concept_bottleneck(image_features_norm=image_features_norm, concept_list=concept_reference_dict.keys() , prompt_info=prompt_info, concept_reference_dict=concept_reference_dict)
        elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
            scores = []
            for concept in concept_reference_dict.keys():
                prompt_template = "{} pattern presented in image"
                text_input = []
                text_input.extend([prompt_template.format(concept)])
                text_input.extend([prompt_template.format(term) for term in concept_reference_dict[concept]])
                scores.append(model.calculate_similarity(img_batch=imgs, text_batch=text_input))
        #elif model_name == "Explicd":
            #predicted_concepts, _ = model.get_concept_predictions(batch=batch, config=config) 
        elif model_name == "Explicd":
            predicted_concepts, _, refinement_info = model.get_concept_predictions_with_self_refine(
                     batch=batch,
                     config=config,
                     use_self_refine=True,
                     llm_refiner=mmed_refiner,
                     margin_threshold=margin_threshold
                )
                    
            # ── save refinement stats for this image ──────────────────────
            dict_refinement_stats[img_ids[0]] = {
                'initial_violations': refinement_info['initial_violations'],
                'final_violations':   refinement_info['final_violations'],
                'converged':          refinement_info['converged'],
                'iterations':         refinement_info['iterations']
            }

        if not raw_values:
            if model_name == "MONET":
                mapped_values = list(map(lambda x: 1 if x > 0.5 else 0, scores))

                # Get dermoscopic concepts
                clinical_concepts = convert_numbers_to_concepts(mapped_values, concept_reference_dict=concept_reference_dict)

                # Create template report
                report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
            elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
                mapped_values = []
                for val in scores:
                    mapped_values.append(val.argmax())

                # Get dermoscopic concepts
                clinical_concepts = convert_numbers_to_concepts(mapped_values)

                # Create template report
                report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
            elif model_name == "Explicd":
                report_template = predicted_concepts + f" Thus the diagnosis is {map_label_to_name(y_true)}."
        else:
            if model_name == "MONET":
                report_template = {
                    "Asymmetry": scores[0],
                    "Irregular": scores[1],
                    "Black": scores[2],
                    "Blue": scores[3],
                    "White": scores[4],
                    "Brown": scores[5],
                    "Erosion": scores[6],
                    "Multiple Colors": scores[7],
                    "Tiny": scores[8],
                    "Regular": scores[9], 
                }
            else:
                report_template = {
                    "Asymmetry": scores[0][:,:1],
                    "Irregular": scores[1][:,:1],
                    "Black": scores[2][:,:1],
                    "Blue": scores[3][:,:1],
                    "White": scores[4][:,:1],
                    "Brown": scores[5][:,:1],
                    "Erosion": scores[6][:,:1],
                    "Multiple Colors": scores[7][:,:1],
                    "Tiny": scores[8][:,:1],
                    "Regular": scores[9][:,:1], 
                }

        dict_to_save_data[img_ids[0]] = str(report_template)
    
    if predict_for_train_set:
        for batch in tqdm(train_dataloader):
            img_ids = batch["img_id"]
            y_true = batch["class_label"].numpy()
            imgs = [Image.open(x) for x in batch["img_path"]]

            if model_name != "Explicd":
                image_embedding = model.extract_image_features(imgs)
                image_features_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)

            if model_name == "MONET":
                scores = model.get_concept_bottleneck(image_features_norm=image_features_norm, concept_list=concept_reference_dict.keys() , prompt_info=prompt_info, concept_reference_dict=concept_reference_dict)
            elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
                scores = []
                for concept in concept_reference_dict.keys():
                    prompt_template = "{} pattern presented in image"
                    text_input = []
                    text_input.extend([prompt_template.format(concept)])
                    text_input.extend([prompt_template.format(term) for term in concept_reference_dict[concept]])
                    scores.append(model.calculate_similarity(img_batch=imgs, text_batch=text_input))
            #elif model_name == "Explicd":
                #predicted_concepts, _ = model.get_concept_predictions(batch=batch, config=config) 
            elif model_name == "Explicd":
                 predicted_concepts, _, refinement_info = model.get_concept_predictions_with_self_refine(   
                     batch=batch,
                     config=config,
                     use_self_refine=True,
                     llm_refiner=mmed_refiner,
                     margin_threshold=margin_threshold 
                )
                 # ── save refinement stats for this train image ────────────
                 dict_refinement_stats[img_ids[0]] = {
                    'initial_violations': refinement_info['initial_violations'],
                    'final_violations':   refinement_info['final_violations'],
                    'converged':          refinement_info['converged'],
                    'iterations':         refinement_info['iterations']
                }

            if not raw_values:
                if model_name == "MONET":
                    mapped_values = list(map(lambda x: 1 if x > 0.5 else 0, scores))

                    # Get dermoscopic concepts
                    clinical_concepts = convert_numbers_to_concepts(mapped_values, concept_reference_dict=concept_reference_dict)

                    # Create template report
                    report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
                elif (model_name == "BiomedCLIP") or (model_name == "CLIP"):
                    mapped_values = []
                    for val in scores:
                        mapped_values.append(val.argmax())

                    # Get dermoscopic concepts
                    clinical_concepts = convert_numbers_to_concepts(mapped_values)

                    # Create template report
                    report_template = generate_template(map_label_to_name(y_true), clinical_concepts)
                elif model_name == "Explicd":
                    report_template = predicted_concepts + f" Thus the diagnosis is {map_label_to_name(y_true)}."
            else:
                if model_name == "MONET":
                    report_template = {
                        "Asymmetry": scores[0],
                        "Irregular": scores[1],
                        "Black": scores[2],
                        "Blue": scores[3],
                        "White": scores[4],
                        "Brown": scores[5],
                        "Erosion": scores[6],
                        "Multiple Colors": scores[7],
                        "Tiny": scores[8],
                        "Regular": scores[9], 
                    }
                else:
                    report_template = {
                        "Asymmetry": scores[0][:,:1],
                        "Irregular": scores[1][:,:1],
                        "Black": scores[2][:,:1],
                        "Blue": scores[3][:,:1],
                        "White": scores[4][:,:1],
                        "Brown": scores[5][:,:1],
                        "Erosion": scores[6][:,:1],
                        "Multiple Colors": scores[7][:,:1],
                        "Tiny": scores[8][:,:1],
                        "Regular": scores[9][:,:1], 
                    }

            dict_to_save_data[img_ids[0]] = str(report_template)

    # Save reports into CSV file
    pre_df = pd.DataFrame.from_dict(dict_to_save_data, orient='index', columns=['Column 2']).reset_index()
    pre_df.columns = ["image_id", "report"]
    if split is None:
            file_path = f"results/concept_prediction/{dataset}_dermatology_reports_generated_by_{model_name}_refiner_{refiner_name}_raw_values_{raw_values}.csv"
    else:
            file_path = f"results/concept_prediction/{dataset}_split_{split}_dermatology_reports_generated_by_{model_name}_refiner_{refiner_name}_raw_values_{raw_values}.csv"

    # Extract the directory path from the file path
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    pre_df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")

    # ── save refinement statistics CSV ────────────────────────────────────────
    if dict_refinement_stats and model_name == "Explicd":
        stats_df = pd.DataFrame.from_dict(
            dict_refinement_stats, orient='index'
        ).reset_index()
        stats_df.columns = [
            'image_id', 'initial_violations', 'final_violations',
            'converged', 'iterations'
        ]
        if split is None:
            stats_path = (
                f"results/concept_prediction/{dataset}_refinement_stats"
                f"_{model_name}_refiner_{refiner_name}.csv"
            )
        else:
            stats_path = (
                f"results/concept_prediction/{dataset}_split_{split}"
                f"_refinement_stats_{model_name}_refiner_{refiner_name}.csv"
            )
        stats_df.to_csv(stats_path, index=False)
        print(f"Refinement stats saved to {stats_path}")
    # ─────────────────────────────────────────────────────────────────────────

    # free GPU memory
    del model
    del test_dataloader
    del dict_to_save_data
    gc.collect()
    torch.cuda.empty_cache()


def build_pred_path(dataset, split, model_name, raw_values, ground_truth_concepts,
                    concept_extractor, n_demos, refiner_name, random_demos, no_hint):
    """Single source of truth for label_prediction CSV filenames."""
    retrieval_tag = "random" if random_demos else "rices"
    hint_tag = "_nohint" if no_hint else ""
    split_tag = f"_split_{split}" if split is not None else ""
    return (
        f"results/label_prediction/{dataset}{split_tag}_{model_name}"
        f"_diagnostic_report_validation_raw_values_{raw_values}"
        f"_gt_concepts_{ground_truth_concepts}"
        f"_model_extractor_{concept_extractor}"
        f"_n_demos_{n_demos}_refiner_{refiner_name}"
        f"_retrieval_{retrieval_tag}{hint_tag}.csv"
    )

def c_to_y(model_name: str, dataset:str, ckpt:str, split=None, raw_values=False, 
           concept_extractor:str=None, report_path: str = None, use_demos=False, 
           n_demos=0, ground_truth_concepts=False, refiner_name:str='mmed', 
           random_demos=False, no_hint=False, model=None):
    """
    Report template:
    > The lesion is diagnosed as {label}. The presence of {", ".join(item for item in concepts)} are highly suggestive of {label}.
    """

    # Load reports
    if dataset == 'PH2':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:  
            df_reports = pd.read_csv(f"results/concept_prediction/PH2_split_{split}_dermatology_reports_generated_by_{concept_extractor}_refiner_{refiner_name}_raw_values_{raw_values}.csv")
        
        PH2_TEST = pd.read_csv(f"/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/PH2/splits/PH2_test_split_{split}.csv")
        PH2_TRAIN = pd.read_csv(f"/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/PH2/splits/PH2_train_split_{split}.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(PH2_TEST.images.to_list())]
        df_reports_train = df_reports.loc[df_reports.image_id.isin(PH2_TRAIN.images.to_list())]
        df_reports_gt = df_reports
    elif dataset == 'Derm7pt':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:
            df_reports = pd.read_csv(f"results/concept_prediction/Derm7pt_dermatology_reports_generated_by_{concept_extractor}_refiner_{refiner_name}_raw_values_{raw_values}.csv")
        
        D7_TEST = pd.read_csv("/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/Derm7pt/splits/derm7pt_test.csv")
        D7_TRAIN = pd.read_csv("/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/Derm7pt/splits/derm7pt_train.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(D7_TEST.images.to_list())]
        df_reports_train = df_reports.loc[df_reports.image_id.isin(D7_TRAIN.images.to_list())]
        if use_demos:
            gt_path = f"results/concept_prediction/Derm7pt_dermatology_reports_generated_by_{concept_extractor}_refiner_{refiner_name}_raw_values_{raw_values}.csv"
            if os.path.exists(gt_path):
                df_reports_gt = pd.read_csv(gt_path)
            else:
                print(f"WARNING: Derm7pt GT reports file not found at {gt_path}, disabling demos.")
                df_reports_gt = None
                use_demos = False
        else:
            df_reports_gt = None
    #    
    elif dataset == 'HAM10000':
        if report_path is not None:
            df_reports = pd.read_csv(report_path) 
        else:
            df_reports = pd.read_csv(f"results/concept_prediction/HAM10000_dermatology_reports_generated_by_{concept_extractor}_refiner_{refiner_name}_raw_values_{raw_values}.csv")

        HAM_TEST = pd.read_csv("/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/HAM10000/splits/HAM10000_test.csv")
        HAM_TRAIN = pd.read_csv("/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/HAM10000/splits/HAM10000_train.csv")
        HAM_VAL = pd.read_csv("/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/HAM10000/splits/HAM10000_val.csv")
        df_reports_test = df_reports.loc[df_reports.image_id.isin(HAM_TEST.image_id.to_list())]
        df_reports_train = pd.concat([df_reports.loc[df_reports.image_id.isin(HAM_TRAIN.image_id.to_list())], df_reports.loc[df_reports.image_id.isin(HAM_VAL.image_id.to_list())]])
        if use_demos:
            gt_path = f"results/concept_prediction/HAM10000_dermatology_reports_generated_by_{concept_extractor}_refiner_{refiner_name}_raw_values_{raw_values}.csv"
            if os.path.exists(gt_path):
                df_reports_gt = pd.read_csv(gt_path)
            else:
                print(f"WARNING: HAM10000 GT reports file not found at {gt_path}, disabling demos.")
                df_reports_gt = None
                use_demos = False
        else:
            df_reports_gt = None 

    #
    else:
        raise ValueError(f"The dataset {dataset} is not implemented.")

    
    # Evaluate — reuse a pre-built model if the caller supplied one (sweep mode)
    if model is None:
        model = build_classifier(model_name, ckpt)

    dict_responses = {
        'image_id': [],
        'gt_response': [],
        'llm_response': [],
        'demonstrations': [],
        'predicted_concepts': []
    }

    # Define instruction and query
    instruction = f"You're a english doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations."
    hint = """Consider the following examples:\n
    A skin lesion is a nevus when it has the majority of the following concepts: uniformly tan, brown, or black, round, sharp and well-defined, regular pigment network, symmetric dots and globules, smooth, symmetrical, raised with possible central ulceration.\n
    A skin lesion is a melanoma when it has the majority of the following concepts: highly variable, often with multiple colors (black, brown, red, white, blue), irregular, often blurry and irregular, atypical pigment network, irregular streaks, blue-whitish veil, irregular, a raised or ulcerated surface, asymmetrical, flat to raised.\n"""
    query = """###Question: What is the type of skin lesion that is associated with the following dermoscopic concepts: {}. ###Options: A. Nevus\nB. Melanoma. ###Answer:"""
    # Inject the decision criteria when there are no demos to anchor the model.
    # MMed survives 0-shot (medical model); general Mistral does not, and
    # collapses to "predict melanoma for everything" (Sens 100 / Spec 0).
    if (not use_demos or n_demos == 0) and not no_hint:
        instruction = instruction + "\n" + hint
    # Demonstrations
    if use_demos:
        if random_demos:
            
            all_train_ids = df_reports_train.image_id.to_list()
        else:
            rices = RICES(dataset=dataset, split=split, valid_ids=[]) # feature_extractor="explicd",
    demos_to_use_in_prompt = None
    n_unparsed = 0
    for img_id, report in tqdm(zip(df_reports_test.image_id.to_list(), df_reports_test.report.to_list())):

        # Demonstrations
        if use_demos:
            if random_demos:
                # Random selection instead of similarity-based
                candidate_ids = rnd.sample(all_train_ids,
                                           min(n_demos * 10, len(all_train_ids)))
                demos_ids = candidate_ids
            else:
                demos_ids = rices.get_context_keys(key=img_id, n=n_demos * 5)
            demos_to_use_in_prompt = []
            # ── Balanced class demo selection ────────────────────────────
            # Collect clean demos separately per class, then interleave
            # so the LLM sees equal evidence for nevus and melanoma.
            clean_demos_by_class = {'nevus': [], 'melanoma': []}
            for id in demos_ids:
                sample = df_reports_train[df_reports_train.image_id == id].report.to_list()
                if len(sample) == 0:
                    continue
                demo_report = sample[0]
                if "Thus the diagnosis is" not in demo_report:
                    continue  # skip if label marker not found

                label_start = demo_report.find("Thus the diagnosis is ") + len("Thus the diagnosis is ")
                demo_label = demo_report[label_start:].strip().rstrip('.').lower()
                demo_concepts_only = demo_report[:demo_report.find("Thus the diagnosis is")-1].strip()

                # Hard safety: skip if any label word survived stripping
                if 'nevus' in demo_concepts_only.lower() or 'melanoma' in demo_concepts_only.lower():
                    continue

                if demo_label not in clean_demos_by_class:
                    continue

                if count_violations(demo_concepts_only) <= 1:
                   demo_letter = 'A' if demo_label == 'nevus' else 'B'
                   exemplar = f"{query.format(demo_concepts_only)} {demo_letter}. {demo_label.capitalize()}"
                   clean_demos_by_class[demo_label].append(exemplar)

                # Stop early once we have enough of both classes
                per_class_needed = max(1, n_demos // 2)
                if (len(clean_demos_by_class['nevus']) >= per_class_needed and
                        len(clean_demos_by_class['melanoma']) >= per_class_needed):
                    break

            # Interleave nevus and melanoma demos (nevus first to avoid recency bias)
            per_class = max(1, n_demos // 2)
            nevus_demos    = clean_demos_by_class['nevus'][:per_class]
            melanoma_demos = clean_demos_by_class['melanoma'][:per_class]
            # zip interleaves; chain picks up any leftover from the longer list
            
            interleaved = [d for pair in zip_longest(nevus_demos, melanoma_demos)
               for d in pair if d is not None]
            if not interleaved:
                # fallback: take any demos regardless of class balance
                all_clean = clean_demos_by_class['nevus'] + clean_demos_by_class['melanoma']
                interleaved = all_clean[:n_demos]
            demos_to_use_in_prompt = interleaved if interleaved else None
    
        if concept_extractor != "Explicd":
            concepts = report[report.find("The presence"):report.find("are highly")-1]
            input_query = query.format(concepts)
            gt_response = report[len("The lesion is diagnosed as "):report.find(". The")]
        else:
            # ── strip label suffix for ALL models, not just MMed ──────────
            # The report format is: "<concepts>. Thus the diagnosis is <label>."
            # We must NEVER let the label leak into the query regardless of LLM.
            diag_marker = "Thus the diagnosis is"
            if diag_marker in report:
                concepts = report[:report.find(diag_marker)-1].strip().rstrip('.')
                gt_response = report[report.find(diag_marker+" ")+len(diag_marker+" "):-1].strip().rstrip('.')
            else:
                concepts = report.strip()
                gt_response = "nevus"  # fallback, should never happen
            input_query = query.format(concepts)
        
        if model_name in ("GPT", "Gemini"):
            raw_output = model.inference_text(
                instruction=instruction,
                query=input_query,
                max_new_tokens=16,
                demos=demos_to_use_in_prompt,
            ).strip()
        else:
            prompt = model.get_prompt(instruction, input_query, demos=demos_to_use_in_prompt)
            raw_output = model.predict(prompt, max_new_tokens=10).strip()

        print(f"[DEBUG] img={img_id} | model={model_name} | raw_output={repr(raw_output)}")
        import re as _re
        match = _re.search(r'\b([AB])\b', raw_output)
        if match:
            llm_response = map_letter_to_label(match.group(1))
        elif 'melanoma' in raw_output.lower():
            llm_response = 'melanoma'
        elif 'nevus' in raw_output.lower():
            llm_response = 'nevus'
        else:
            n_unparsed += 1
            llm_response = 'nevus'   # consistent, conservative default
            
            
        dict_responses['image_id'].append(img_id)
        dict_responses['gt_response'].append(gt_response)
        dict_responses['llm_response'].append(llm_response)
        # NOTE: DEBUG
        dict_responses['demonstrations'].append(demos_to_use_in_prompt)
        dict_responses['predicted_concepts'].append(concepts)


    print(f"[PARSE] {dataset} {model_name} refiner={refiner_name} "
          f"n_demos={n_demos}: {n_unparsed} unparsed answers")

    # Converter para DataFrame
    df = pd.DataFrame(dict_responses)
    
    file_path = build_pred_path(dataset, split, model_name, raw_values,
                                ground_truth_concepts, concept_extractor,
                                n_demos, refiner_name, random_demos, no_hint)
    
    # Extract the directory path from the file path
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def build_classifier(model_name: str, ckpt: str):
    """Single place where a Stage-3 classifier is instantiated."""
    if model_name == "MMed":
        return MMedLlama3(ckpt)
    elif model_name == "Mistral":
        return Mistral()
    elif model_name == "MedGemma":
        return MedGemma(ckpt=ckpt)
    elif model_name == "Qwen":
        return Qwen(ckpt=ckpt)
    elif model_name == "GPT":
        return GPT5(model=ckpt)
    elif model_name == "Gemini":
        return Gemini(model=ckpt)
    raise TypeError(f"The specified model {model_name} does not have a valid implementation.")

def classification(model_name: str, dataset:str, ckpt:str, split=None, raw_values=False, concept_extractor:str=None, report_path: str = None, use_demos=False, n_demos=0, ground_truth_concepts=False, refiner_name:str='mmed', random_demos=False, no_hint=False):

    file_path = build_pred_path(dataset, split, model_name, raw_values,
                                ground_truth_concepts, concept_extractor,
                                n_demos, refiner_name, random_demos, no_hint)
    df_responses = pd.read_csv(file_path)
    if dataset == "PH2":
        PH2_TEST = pd.read_csv(f"/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/PH2/splits/PH2_test_split_{split}.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(PH2_TEST.images.to_list())]
    elif dataset == "Derm7pt":
        D7_TEST = pd.read_csv("/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/Derm7pt/splits/derm7pt_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(D7_TEST.images.to_list())]
    elif dataset == "HAM10000":
        HAM_TEST = pd.read_csv("/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/data/HAM10000/splits/HAM10000_test.csv")
        df_filtered = df_responses.loc[df_responses.image_id.isin(HAM_TEST.image_id.to_list())]

    mapping = {
        'nevus': 0,
        'melanoma': 1,
    }
    
    y_true = df_filtered.gt_response.map(mapping).to_list()
    y_pred = df_filtered.llm_response.map(mapping).to_list()
    
    # Get results
    results = calculate_metrics(y_true, y_pred)

    # Save results to JSON
    save_data_to_json(results, model=model_name, subdir='x_to_c_to_y', dataset=dataset, split=split, task=f"gt_concepts_{ground_truth_concepts}_raw_values_{raw_values}_model_extractor_{concept_extractor}_n_demos_{n_demos}")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concept to Class label Classification')
    parser.add_argument('--model', type=str, default='Explicd')
    parser.add_argument('--dataset', type=str, default='Derm7pt')
    parser.add_argument('--report_path', type=str, default=None)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--raw_values', action="store_true")
    parser.add_argument('--ckpt', type=str, default='/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/checkpoint/MMed-Llama-3-8B-EnIns')
    parser.add_argument('--classifier_ckpt', type=str, default=None, help='Checkpoint for classifier LLM (if different from refiner ckpt)')
    parser.add_argument('--concept_extractor', type=str, default='Explicd')
    parser.add_argument('--concept_reference_dict', type=str, default='PH2')
    parser.add_argument('--llm', type=str, default=None,
                        choices=['MMed', 'Mistral', 'Qwen', 'MedGemma'],
                        help='Stage-3 classifier.')
    parser.add_argument('--use_demos', action="store_true")
    parser.add_argument('--predict_for_train_set', action="store_true")
    parser.add_argument('--n_demos', type=int, default=0)
    parser.add_argument('--gt_concepts', action="store_true")
    parser.add_argument('--generate_concepts', action="store_true")
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--refiner', type=str, default='mmed',
                        choices=['rule', 'mistral', 'mmed', 'none'],
                        help='Concept refiner arm. ONLY rule/mistral/mmed are valid — '
                             'Qwen and MedGemma are classifier-only models.')
    parser.add_argument('--refiner_list', nargs='+', default=None,
                        choices=['rule', 'mistral', 'mmed', 'none'],
                        help='Sweep several refiner arms in ONE process (model loaded once).')
    parser.add_argument('--split_list', nargs='+', type=int, default=None,
                        help='Sweep several PH2 splits in ONE process.')
    parser.add_argument('--random_demos', action="store_true", help='Use random demo selection instead of RICES')
    parser.add_argument('--no_hint', action="store_true", help='Disable hint in prompt (affects output filename tag)')
    parser.add_argument('--margin_threshold', type=float, default=0.2)
    args = parser.parse_args()
    seed_everything(seed=42)

    print("\n" + "#"*80)
    print(f"# Status: Running...")
    print(f"# LLM          : {args.llm}")
    print(f"# Refiner      : {args.refiner}")
    print(f"# Dataset      : {args.dataset}")
    print(f"# n-shots      : {args.n_demos}")
    print(f"# Generate X→C : {args.generate_concepts}")
    print(f"# Date         : {get_current_date()}")
    print("#"*80)

    classifier_ckpt = args.classifier_ckpt if args.classifier_ckpt else args.ckpt

    # ====================== STEP 1: Generate Concepts (X → C) ======================
    if args.generate_concepts:
        x_to_c(
            model_name=args.model,
            dataset=args.dataset,
            concept_reference_dict=args.concept_reference_dict,
            split=args.split,
            raw_values=args.raw_values,
            predict_for_train_set=args.predict_for_train_set,
            data_path=args.data_path,
            refiner_name=args.refiner,
            margin_threshold=args.margin_threshold
        )

    # ====================== STEP 2 + 3: Classify and Evaluate ======================
    else:
        refiners = args.refiner_list or [args.refiner]
        splits   = args.split_list   or [args.split]

        # Build the classifier ONCE, reuse across the whole sweep.
        clf_model = build_classifier(args.llm, classifier_ckpt)

        for _ref in refiners:
            for _sp in splits:
                print(f"\n=== classifier={args.llm} refiner={_ref} split={_sp} "
                      f"n_demos={args.n_demos} ===")

                c_to_y(
                    model_name=args.llm,
                    dataset=args.dataset,
                    ckpt=classifier_ckpt,
                    split=_sp,
                    raw_values=args.raw_values,
                    concept_extractor=args.concept_extractor,
                    report_path=args.report_path,
                    use_demos=args.use_demos,
                    n_demos=args.n_demos,
                    ground_truth_concepts=args.gt_concepts,
                    refiner_name=_ref,
                    random_demos=args.random_demos,
                    no_hint=args.no_hint,
                    model=clf_model,
                )

                classification(
                    model_name=args.llm,
                    dataset=args.dataset,
                    ckpt=classifier_ckpt,
                    split=_sp,
                    ground_truth_concepts=args.gt_concepts,
                    raw_values=args.raw_values,
                    concept_extractor=args.concept_extractor,
                    n_demos=args.n_demos,
                    refiner_name=_ref,
                    random_demos=args.random_demos,
                    no_hint=args.no_hint,
                )

    print("\n" + "#"*80)
    print(f"# Status: Finished Successfully!")
    print(f"# Date: {get_current_date()}")
    print("#"*80)
    print("\n")
