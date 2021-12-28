# Copyright Xinapse NLU_Data team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
import torch
from torchmetrics.functional import auroc
from transformers import AdamW, AutoModelForMaskedLM, AutoConfig
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import pickle


class VOC_TopicLabeler(pl.LightningModule):
    """
    Multiclass Classification model for SEMA Group's VOC Sentiment Trend Analysis
    """
    def __init__(self, n_classes: int = 16,
                 n_training_steps=None,
                 n_warmup_steps=None,
                 config=None):
        """
        Args:
            :param n_classes: Number of classes to classify
            :param n_training_steps: Number of training steps, not used for inferencing
            :param n_warmup_steps: Number of Warmup steps: not used for inferencing
        """
        super().__init__()
        self.config = config
        # self.config = AutoConfig.from_pretrained("klue/roberta-base", output_hidden_states=True)
        # self.config.max_position_embeddings = 512
        self.model = AutoModelForMaskedLM.from_pretrained("klue/roberta-base", config=self.config)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCEWithLogitsLoss()  # nn.BCELoss() with sigmoid layer
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.activation = nn.Tanh()

        with open('pickles/data.pkl', 'rb') as f:
            mlb = pickle.load(f)
        self.LABEL_COLUMNS = mlb.classes_[:]

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = output.hidden_states[-1]
        pooled_output = self.classifier(self.dropout(self.activation(self.dense(last_hidden_state[:, 0]))))
        loss = 0
        if labels is not None:
            loss = self.criterion(pooled_output, labels)
        return loss, torch.sigmoid(pooled_output)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataset_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("predic_loss", loss, prog_bar=True, logger=True)
        return loss, outputs

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        for i, name in enumerate(self.LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))
