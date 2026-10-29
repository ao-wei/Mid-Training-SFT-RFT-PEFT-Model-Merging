import torch
import torch.nn as nn
import math

class LinearLoRALayer(nn.Module):
    def __init__(self,
            input_dim,
            output_dim,
            rank=8,
            lora_alpha=16,
            dropout=0.1,
            merge=False,
            debug=False,
                 ):
        super().__init__()

        assert rank > 0, "rank must be larger than zero"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge = merge
        self.rank = rank

        # 注意这里的self.linear.weight.data shape是[output_dim, input_dim], X*W^T
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        # 定义LoRA可学习低秩矩阵 B, shape=[output_dim, rank]
        self.lora_b = nn.Parameter(torch.zeros(output_dim, rank))

        # 定义LoRA可学习低秩矩阵 A, shape=[rank, input_dim]
        # 记住一点，给输入 X 下投影那个矩阵要随机初始化，上投影那个矩阵要全0初始化
        self.lora_a = nn.Parameter(torch.zeros(rank, input_dim))
        nn.init.normal_(self.lora_a, mean=0, std=0.01)

        if debug:
            print("lora_a initial value: ", self.lora_a)
            print("lora_b initial value: ", self.lora_b)

        # 定义delta权重更新的系数
        self.scale = lora_alpha / rank

        # 冻结 linear weight，不可训练
        self.linear.weight.requires_grad = False

        # 定义dropout
        self.dropout = nn.Dropout(dropout)

        # 如果要做推理，而不是训练，一般都要做 merge， 加快推理速度
        if merge:
            self.merge_lora()

        def forward(self, hidden_states):
            # hidden_states shape = [batch, seq_len, input_dim]
            # lora_b shape = [output_dim, rank]
            # lora_a shape = [rank, input_dim]
            # lora_b * lora_a = [output_dim, input_dim]
            # output: hidden_states * W^T = [batch, seq_len, input_dim] * [input_dim, output_dim] = [batch, seq_len, output_dim] 
            if self.merge:
                output = self.linear(hidden_states)
            else:
                output = self.linear(hidden_states) + self.scale * (hidden_states @ (self.lora_b @ self.lora_a).T)

            return self.dropout(output)

        def merge_lora(self):
            self.linear.weight.data += self.scale * (self.lora_b @ self.lora_a)

        def unmerge_lora(self):
            self.linear.weight.data -= self.scale * (self.lora_b @ self.lora_a) 