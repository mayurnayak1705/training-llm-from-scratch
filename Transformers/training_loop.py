from scratch import *
from training_llm import *
import torch
import numpy as np



device_mps = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(device_mps)
tokenizer = BPE_Tokenizer_Regex()
tokenizer.load("/home/Transformers/bpe_tokenizer.json")
encoded = tokenizer.encode("hello world")
encoded_np = np.array(encoded, dtype=np.int32)
decoded = tokenizer.decode(encoded)

# corpus = ""
# with open("/home/Transformers/DATA/tinystories.txt", "r") as f:
#     for line in f:
#         corpus += line.strip() + " " 
# corpus = corpus[0:2000000]

# print(len(corpus))

# chunk_size = 100000
# encoded_data_list = []

# for i in range(0, len(corpus), chunk_size):
#     print(i)
#     chunk = corpus[i:i+chunk_size]
#     encoded_chunk = tokenizer.encode(chunk)
#     encoded_data_list.extend(encoded_chunk)

# encoded_data_np = np.array(encoded_data_list, dtype=np.int32)
# np.save("DATA/encoded_data.npy", encoded_data_np)
# print(f"Total tokens: {len(encoded_data_np)}")


encoded_data_np = np.load("DATA/encoded_data.npy")
# train, labels = data_loading(encoded_data_np, 10, 100, device_mps)
# train, labels = train.to(device_mps), labels.to(device_mps)

def training_loop(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000.0,lr=1e-3, beta1=0.9 ,beta2 = 0.999, eps=1e-8, weight_decay=0.01,max_grad_norm=1.0,total_steps=251, warmup_steps=50):
    betas = (beta1,beta2)

    model = Transformer_LM(vocab_size, context_length,d_model,num_layers,num_heads, d_ff,rope_theta)
    model = model.to(device_mps)
    lossfn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


    for step in range(total_steps):
        train, labels = data_loading(encoded_data_np, batch_size=8, context_length=200, device=device_mps)
        lr_step = get_lr_cosine_schedule(step, alpha_max=lr, alpha_min=lr*0.1, Tw=warmup_steps, Tc=total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_step
        logits = model(train)
        batch, seq, vocab = logits.size()
        logits = logits.view(batch*seq, vocab)
        loss = lossfn(logits, labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)
        optimizer.step()
        print(f"loss on iteration {step} is {loss}")

        if step % 10 == 0:
            save_checkpoint(model, optimizer, step, out="/home/Transformers/checkpoints/latest_checkpint.pthß")

training_loop(3000, 300, 32, 8, 4, 128)



import torch
import torch.nn.functional as F

def generate_text(model, start_tokens, max_new_tokens=200, temperature=0.9, top_k=None, device="cpu"):

    model.eval()
    tokens = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)  # shape [1, seq_len]

    for _ in range(max_new_tokens):
        if tokens.size(1) > model.context_length:
            tokens = tokens[:, -model.context_length:]

        with torch.no_grad():
            logits = model(tokens)  
        
        logits = logits[:, -1, :] 
        logits = logits / temperature
        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens.squeeze(0).tolist() 


model = Transformer_LM(vocab_size=3000, context_length=300, d_model=32, num_layers=8, num_heads=4, d_ff=128,rope_theta=10000.0)
model = model.to(device_mps)

iteration = load_checkpoint("/home/Transformers/checkpoints/latest_checkpint.pthß", model,)
print(f"Loaded model from iteration {iteration}")


model.eval()
start_tokens = [10, 25, 47] 
generated = generate_text(model, start_tokens, max_new_tokens=300, temperature=0.8, top_k=50, device=device_mps)
print("Generated tokens:", generated)
print(tokenizer.decode(generated))