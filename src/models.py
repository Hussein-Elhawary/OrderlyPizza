import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from utils import *

class PizzaOrderDataset(Dataset):
    def __init__(self, words_list, labels_list, tokenizer, word2vec_model, tfidf_vectorizer):
        self.words_list = words_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model
        self.tfidf_vectorizer = tfidf_vectorizer

        # Calculate max_length as the length of the longest list in words_list
        self.max_length_words = max(len(words) for words in words_list)
        self.max_length_labels = len(labels_list)
        print(self.max_length_words)
        print(self.max_length_labels)
        # Create label-to-index mapping
        self.label_to_index = self._create_label_mapping()
        print(self.label_to_index)
        print(labels_list)
    
    def _create_label_mapping(self):
        #unique_labels = set(label for labels in self.labels_list for label in labels)
        return {label: idx for idx, label in enumerate(self.labels_list)}
    
    def __len__(self):
        return len(self.words_list)
    
    def __getitem__(self, idx):
        # Prepare words and labels
        words = self.words_list[idx]
        labels = self.labels_list
        
        # Tokenize
        encodings = self.tokenizer(words, is_split_into_words=True, 
                                   padding='max_length', 
                                   truncation=True, 
                                   max_length=self.max_length_words)
        print(encodings)
        # Word2Vec indices
        word_indices = torch.tensor([
            self.word2vec_model.wv.key_to_index.get(word, 0) 
            for word in words
        ])
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([' '.join(words)]).toarray()[0]
        tfidf_tensor = torch.tensor(tfidf_features, dtype=torch.float32)
        
        # Convert labels to indices
        label_indices = torch.tensor([
            self.label_to_index[label] for label in labels
        ])
        
        # Pad word_indices and label_indices to max_length
        word_indices = torch.nn.functional.pad(word_indices, (0, self.max_length_words - len(word_indices)), value=-1)
        label_indices = torch.nn.functional.pad(label_indices, (0, self.max_length_labels - len(label_indices)), value=-100)
        
        print("words indices",word_indices)
        print("labels indices",label_indices)
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'word_indices': word_indices,
            'tfidf_features': tfidf_tensor,
            'labels': label_indices
        }

class PizzaOrderNERModel(nn.Module):
    def __init__(self, 
                 roberta_model='roberta-base', 
                 num_labels=13,
                 word2vec_dim=100,
                 tfidf_dim=1000):
        super(PizzaOrderNERModel, self).__init__()
        
        # RoBERTa contextual embeddings
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        
        # Word2Vec embedding layer
        self.word_embedding_layer = nn.Embedding(
            num_embeddings=len(word2vec_model.wv.key_to_index), 
            embedding_dim=word2vec_dim
        )
        
        # TF-IDF projection layer
        self.tfidf_projection = nn.Linear(tfidf_dim, 768)  # Project to RoBERTa embedding size
        
        # Combine embeddings
        total_embedding_dim = 768 + word2vec_dim + 768
        
        # Classification layers
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(total_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, word_indices, tfidf_features):
        # Get RoBERTa contextual embeddings
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        contextual_embeddings = roberta_output.last_hidden_state
        
        # Get Word2Vec embeddings
        word_embeddings = self.word_embedding_layer(word_indices)
        
        # Get TF-IDF embeddings and project to RoBERTa embedding size
        tfidf_embeddings = self.tfidf_projection(tfidf_features)
        
        # Combine all embeddings
        combined_embeddings = torch.cat([
            contextual_embeddings, 
            word_embeddings.unsqueeze(1).expand(-1, contextual_embeddings.size(1), -1), 
            tfidf_embeddings.unsqueeze(1).expand(-1, contextual_embeddings.size(1), -1)
        ], dim=-1)
        
        # Classification
        logits = self.classifier(self.dropout(combined_embeddings))
        return logits

def prepare_embeddings(words_list):
    # Train Word2Vec
    global word2vec_model
    word2vec_model = Word2Vec(sentences=words_list, vector_size=100, window=5, min_count=1, workers=4)
    
    # Prepare TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(words) for words in words_list])
    
    return word2vec_model, tfidf_vectorizer, tfidf_matrix

def train_ner_model(words_list, labels_list, epochs=10, batch_size=4, learning_rate=2e-5):
    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    
    # Prepare embeddings
    word2vec_model, tfidf_vectorizer, tfidf_matrix = prepare_embeddings(words_list)
    
    # Create dataset
    dataset = PizzaOrderDataset(words_list, labels_list, tokenizer, word2vec_model, tfidf_vectorizer)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    num_labels = len(labels_list)
    print(num_labels)
    model = PizzaOrderNERModel(num_labels=num_labels, tfidf_dim=tfidf_vectorizer.max_features)  # Ensure tfidf_dim matches the TF-IDF vectorizer
    
    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                word_indices=batch['word_indices'],
                tfidf_features=batch['tfidf_features']
            )
            
            # Reshape logits and labels for loss calculation
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),  # Flatten logits
                batch['labels'].view(-1)  # Flatten labels
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(dataloader)}")
    
    return model

# Example usage
words_list = [
    ['margherita', 'pizza', 'large', 'extra', 'cheese'],
    ['pepperoni', 'medium', 'with', 'olives']
]
labels_path = "./unique_labels.txt"
labels_list = read_labels(labels_path)



# Train the model
model = train_ner_model(words_list, labels_list)