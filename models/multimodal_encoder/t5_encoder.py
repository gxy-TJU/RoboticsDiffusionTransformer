import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5Embedder:
    #available_models = ["google/t5-v1_1-xxl"]
    #available_models = ["google/t5-base"]
    
    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
        local_files_only=False,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir
        print('use_offload_folder', use_offload_folder)
        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
            
            if use_offload_folder is not None:
                print('use_offload_folder', use_offload_folder)
                t5_model_kwargs["offload_folder"] = use_offload_folder
                t5_model_kwargs["device_map"] = {
                    "shared": "disk",
                    "encoder.embed_tokens": "disk",
                    "encoder.block.0": "disk",
                    "encoder.block.1": "disk",
                    "encoder.block.2": "disk",
                    "encoder.block.3": "disk",
                    "encoder.block.4": "disk",
                    "encoder.block.5": "disk",
                    "encoder.block.6": "disk",
                    "encoder.block.7": "disk",
                    "encoder.block.8": "disk",
                    "encoder.block.9": "disk",
                    "encoder.block.10": "disk",
                    "encoder.block.11": "disk",
                    "encoder.block.12": "disk",
                    "encoder.block.13": "disk",
                    "encoder.block.14": "disk",
                    "encoder.block.15": "disk",
                    "encoder.block.16": "disk",
                    "encoder.block.17": "disk",
                    "encoder.block.18": "disk",
                    "encoder.block.19": "disk",
                    "encoder.block.20": "disk",
                    "encoder.block.21": "disk",
                    "encoder.block.22": "disk",
                    "encoder.block.23": "disk",
                    "encoder.final_layer_norm": "disk",
                    "encoder.dropout": "disk",
                }
                
            else:
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder": self.device,
                }

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        #assert from_pretrained in self.available_models
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        
        )
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **t5_model_kwargs,
            #from_tf=True
        ).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,

            max_length=self.model_max_length,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask