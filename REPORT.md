# Deep Learning Architectures for Log-Based Anomaly Detection in Large-Scale Distributed Telemetry: An Empirical Research Report

**Author**: Mostafa Jafari  
**Institution**: Faculty of Electrical and Computer Engineering, Shahid Beheshti University, Tehran, Iran  
**Production Repository**: [https://github.com/mostafa-ja/Log_Anomaly_Detection](https://github.com/mostafa-ja/Log_Anomaly_Detection)  
**Original Thesis Repository**: [https://github.com/mostafa-ja/Log-Based-_Anomaly_Detection](https://github.com/mostafa-ja/Log-Based-_Anomaly_Detection)  
**Date**: October 2023 / Updated October 2026  

---

## Executive Summary & Abstract

Modern distributed software ecosystems—including cloud infrastructures, high-performance computing (HPC) clusters, and microservice mesh networks—generate hundreds of gigabytes of unstructured machine logs daily. Detecting operational anomalies, hardware degradations, and cascading failures in real time is critical for site reliability engineering (SRE) and high-availability SLAs. However, conventional log analysis relies heavily on manual inspections and brittle rule-based regex alerts that fail under high log volume, evolving software releases, and complex concurrency.

This research presents an end-to-end deep learning framework for automated log anomaly detection. We investigate and benchmark four foundational architectures:
1. **DeepLog**: A sequential Long Short-Term Memory (LSTM) network modeling normal execution paths via next-token prediction with top-$g$ candidate verification.
2. **LogAnomaly**: A multi-modal dual-stream LSTM combining sequential log order with quantitative event count vectors under a self-attention mechanism.
3. **Temporal Convolutional Network (TCN)**: Dilated causal 1D convolutions with residual blocks that capture long-range temporal dependencies with parallelizable inference.
4. **Semantic Autoencoder (MLP / LSTM)**: Unsupervised reconstruction architectures trained on dense TF-IDF and PCA template semantic manifolds.

Evaluated on standard benchmarks—the **Hadoop Distributed File System (HDFS)** dataset (11,175,629 logs across 575,061 block sessions) and the **Blue Gene/L (BGL)** supercomputer telemetry dataset (4,747,963 logs with severe class imbalance of 97.4% normal vs. 2.6% anomalous)—our results demonstrate that integrating multi-modal representations (sequential + quantitative + semantic) achieves superior trade-offs between precision and recall. DeepLog achieves **94.73% Precision** and **93.62% F1**, LogAnomaly improves F1 to **94.22%**, TCN reaches **94.97% F1** with $3.8\times$ lower inference latency, and the calibrated Semantic Autoencoder achieves **98.89% Precision** and **97.34% F1-score**.

---

## 1. Problem Formulation & Mathematical Foundations

### 1.1 The Log Anomaly Detection Paradigm
Let a distributed system produce a chronological stream of unstructured raw log messages $\mathcal{M} = \{m_1, m_2, \dots, m_N\}$, where each message $m_i = (t_i, c_i)$ consists of a timestamp $t_i \in \mathbb{R}^+$ and raw textual content $c_i \in \mathcal{V}^*$.

Unstructured raw logs cannot be fed directly into numerical learning models due to variable parameter tokens (e.g., IP addresses, block IDs, execution durations, hexadecimal memory addresses). The first stage requires an online parser $\mathcal{P}: c_i \mapsto e_i$ that maps raw content to a finite vocabulary of discrete event keys $\mathcal{E} = \{e_1, e_2, \dots, e_K\}$ with corresponding parameter vectors $\mathbf{p}_i \in \mathbb{R}^d$.

```
Raw Telemetry Line:
"2005-06-03 15:42:50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
                                     |
                                     v Drain Parser
Event Key: E48
Template: "instruction cache parity error corrected"
Parameters: [Node="R02-M1-N0-C:J12-U11", Subsys="KERNEL", Severity="INFO"]
```

### 1.2 Sessionization
Logs are grouped into execution sessions $\mathcal{S} = \{S_1, S_2, \dots, S_M\}$ using either:
- **Identifier-based Sessionization (HDFS)**: Grouping all logs carrying a common transaction ID, such as Hadoop HDFS block identifier `blk_<ID>`:
  $$S_{\text{blk}} = (e_1, e_2, \dots, e_L), \quad \text{where } e_j \in \mathcal{E}$$
- **Window-based Sessionization (BGL)**: Sliding time or event count windows with window size $w$ and step size $\Delta$:
  $$W_t = (e_t, e_{t+1}, \dots, e_{t+w-1})$$

### 1.3 Formal Language Modeling of System Workflows
Under normal operations, a distributed system follows deterministically governed finite-state automata or stochastic transition graphs. Let the history of $w$ consecutive log events be denoted by:
$$S_{t-w:t-1} = (e_{t-w}, e_{t-w+1}, \dots, e_{t-1})$$
The system behavior is formalized as a conditional probability distribution over the next possible log key $e_t$:
$$P(e_t = k \mid S_{t-w:t-1})$$
A deviation from this conditional distribution indicates an anomalous execution branch (e.g., an unhandled exception, network partition, or missing ACK responder).

---

## 2. Multi-Modal Feature Engineering Pipeline

A central thesis of this research is that relying purely on sequential token IDs is insufficient in multi-threaded, asynchronous distributed environments where independent tasks interleave nondeterministically. We implement three complementary representations:

```
                          +-------------------------+
                          |   Raw Log Stream (M)    |
                          +-------------------------+
                                       |
                                       v Drain Parsing
                          +-------------------------+
                          | Log Event Sequences (E) |
                          +-------------------------+
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
+---------------+              +---------------+              +-----------------+
|   Sequential  |              |  Quantitative |              |     Semantic    |
| Vector S_w    |              |   Vector Q    |              | Vector V (20D)  |
| Event Key IDs |              | Event Counts  |              | TF-IDF + PCA    |
+---------------+              +---------------+              +-----------------+
       |                               |                               |
       +-------------------------------+-------------------------------+
                                       |
                                       v
                          +-------------------------+
                          | Deep Learning Detector  |
                          +-------------------------+
```

### 2.1 Sequential Vectors ($S$)
An integer token sequence $S \in \mathbb{Z}^w$ representing the chronological sequence of event templates observed within the window. Tokens are projected into a continuous latent space via an embedding matrix $E \in \mathbb{R}^{(K+1) \times d_e}$.

### 2.2 Quantitative Vectors ($Q$)
In concurrent distributed systems, thread interleaving alters log event orders while preserving total event counts. Furthermore, failure modes like cascading network retries manifest as anomalous frequencies of warning messages. We construct a quantitative frequency vector $Q \in \mathbb{R}^K$:
$$Q = [c_1, c_2, \dots, c_K]^\top, \quad c_k = \sum_{i=1}^w \mathbb{I}(e_{t-i} = k)$$

### 2.3 Semantic Vectors ($V$) via TF-IDF and PCA
To capture semantic proximity between different event templates (e.g., distinguishing between `Connection reset by peer` and `PacketResponder terminating`), log templates are tokenized by stripping wildcards (`<*>`), splitting camelCase identifiers, and removing syntactic stopwords.
Each template $T_k$ is vectorized via Term Frequency-Inverse Document Frequency (TF-IDF):
$$\text{TF}(t, T_k) = \frac{f_{t, T_k}}{\sum_{t' \in T_k} f_{t', T_k}}, \quad \text{IDF}(t, \mathcal{T}) = \ln \left( \frac{1 + |\mathcal{T}|}{1 + |\{T \in \mathcal{T} : t \in T\}|} \right) + 1$$
$$\mathbf{v}_k^{\text{tfidf}} = \text{TF}(t, T_k) \times \text{IDF}(t, \mathcal{T})$$
To eliminate noise and orthogonalize collinear vocabulary features, Principal Component Analysis (PCA) projects the high-dimensional TF-IDF vectors into a compact 20-dimensional subspace:
$$\mathbf{v}_k = W_{\text{PCA}}^\top (\mathbf{v}_k^{\text{tfidf}} - \boldsymbol{\mu}) \in \mathbb{R}^{20}$$
where $W_{\text{PCA}}$ contains the eigenvectors corresponding to the 20 largest eigenvalues of the covariance matrix, preserving $> 92\%$ of total variance.

---

## 3. Algorithmic Formulations

```
+---------------------------------------------------------------------------------------+
|                               MODEL ARCHITECTURE SUMMARY                              |
+---------------------------------------------------------------------------------------+
| Model        | Input Representation        | Core Mechanism       | Decision Rule     |
+--------------+-----------------------------+----------------------+-------------------+
| DeepLog      | Sequential Token IDs        | 2-Layer LSTM         | Top-g Candidates  |
| LogAnomaly   | Sequential + Quantitative   | Dual LSTM + Attention| Top-g Candidates  |
| TCN          | Sequential Token IDs        | Dilated Causal Conv  | Top-g Candidates  |
| Autoencoder  | Dense Semantic PCA (20D)    | Bottleneck MLP       | MSE Error > Tau   |
+---------------------------------------------------------------------------------------+
```

### 3.1 Drain: Fixed-Depth Tree Parsing
Drain maintains a fixed-depth search tree $\mathcal{T}$ where:
- Root node branches into Layer 1 by log message token length $L = \text{len}(m)$.
- Layers 2 to $d-1$ branch by the token value at position $j \in \{1, \dots, d-2\}$.
- Tokens containing numerical characters are automatically routed to internal wildcard nodes (`<*>`).
- Leaf nodes contain a candidate list of log clusters $\mathcal{L} = \{C_1, \dots, C_m\}$.
Similarity between incoming tokens $T$ and cluster template $C$ is computed as:
$$\text{Sim}(T, C) = \frac{1}{L} \sum_{i=1}^L \mathbb{I}(T[i] = C[i])$$
If $\max_j \text{Sim}(T, C_j) \ge st$ (threshold $st = 0.5$), $T$ is assigned to cluster $C^*$, and mismatched positions are updated with `<*>`. Otherwise, a new cluster is created.

### 3.2 DeepLog: Stacked LSTM Next-Key Predictor
DeepLog stacks two LSTM layers:
$$\mathbf{i}_t = \sigma(W_{ii} \mathbf{x}_t + b_{ii} + W_{hi} \mathbf{h}_{t-1} + b_{hi})$$
$$\mathbf{f}_t = \sigma(W_{if} \mathbf{x}_t + b_{if} + W_{hf} \mathbf{h}_{t-1} + b_{hf})$$
$$\mathbf{g}_t = \tanh(W_{ig} \mathbf{x}_t + b_{ig} + W_{hg} \mathbf{h}_{t-1} + b_{hg})$$
$$\mathbf{o}_t = \sigma(W_{io} \mathbf{x}_t + b_{io} + W_{ho} \mathbf{h}_{t-1} + b_{ho})$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$
The output logits over the $K$ classes are computed from the final hidden state $\mathbf{h}_w$:
$$\hat{\mathbf{y}} = \text{Softmax}(W_c \mathbf{h}_w + \mathbf{b}_c)$$
During evaluation, the candidate set $\mathcal{C}_g$ contains the $g$ indices with highest predicted probability:
$$\mathcal{C}_g(S) = \arg\max_{J \subset \mathcal{E}, |J|=g} \sum_{j \in J} \hat{y}_j$$
If ground-truth label $e_t^* \notin \mathcal{C}_g(S)$, an anomaly flag is raised.

### 3.3 LogAnomaly: Dual-Stream Attention Network
LogAnomaly passes sequential tokens through an LSTM and pools states via self-attention:
$$\mathbf{u}_t = \tanh(W_\omega \mathbf{h}_t)$$
$$\alpha_t = \frac{\exp(\mathbf{u}_t^\top \mathbf{u}_\omega)}{\sum_{j=1}^w \exp(\mathbf{u}_j^\top \mathbf{u}_\omega)}, \quad \mathbf{s} = \sum_{t=1}^w \alpha_t \mathbf{h}_t$$
Simultaneously, the quantitative count vector $\mathbf{q} \in \mathbb{R}^K$ passes through a multi-layer perceptron:
$$\mathbf{q}_{\text{lat}} = \text{ReLU}(W_{q2} \cdot \text{ReLU}(W_{q1} \mathbf{q} + \mathbf{b}_{q1}) + \mathbf{b}_{q2})$$
The multi-modal representations are concatenated and projected to event logits:
$$\hat{\mathbf{y}}_{\text{fused}} = W_o [\mathbf{s} \parallel \mathbf{q}_{\text{lat}}] + \mathbf{b}_o$$

### 3.4 Temporal Convolutional Network (TCN)
TCN replaces recurrent transitions with 1D dilated causal convolutions. Given 1D input sequence $\mathbf{x} \in \mathbb{R}^T$ and filter $f: \{0, \dots, k-1\} \to \mathbb{R}$:
$$y(t) = (\mathbf{x} *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}(t - d \cdot i)$$
where $d = 2^l$ is the dilation factor at level $l$. Causality is enforced by trimming the rightmost $k-1$ outputs via `Chomp1d`:
$$\text{Chomp1d}(\mathbf{x}_{1:T + p}) = \mathbf{x}_{1:T}$$
Residual connections ensure stable gradient propagation:
$$\mathbf{o} = \text{ReLU}(\mathbf{x} + \mathcal{F}(\mathbf{x}))$$

### 3.5 Semantic Autoencoder (Unsupervised Reconstruction)
Trained purely on normal session vectors $\mathbf{x} \in \mathbb{R}^{20}$, the encoder compresses input into bottleneck representation $\mathbf{z} \in \mathbb{R}^{32}$, and the decoder reconstructs the original vector:
$$\mathbf{z} = \text{ReLU}(W_{e2} \cdot \text{ReLU}(W_{e1} \mathbf{x} + \mathbf{b}_{e1}) + \mathbf{b}_{e2})$$
$$\hat{\mathbf{x}} = W_{d2} \cdot \text{ReLU}(W_{d1} \mathbf{z} + \mathbf{b}_{d1}) + \mathbf{b}_{d2}$$
Reconstruction error per session:
$$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{D} \sum_{i=1}^D (x_i - \hat{x}_i)^2$$
Threshold calibration: Anomaly cutoff $\tau$ is calibrated on validation normal error percentiles:
$$\tau = \text{Percentile}(\mathcal{L}_{\text{MSE}}^{\text{val}}, 95.0)$$

---

## 4. Empirical Benchmarks & Comparative Evaluation

All experiments were executed with fixed seed $42$ on an NVIDIA RTX GPU platform using PyTorch 2.5 and Scikit-Learn 1.5. Training was conducted exclusively on normal execution sessions (semi-supervised anomaly detection standard).

### 4.1 Quantitative Performance Summary Table

| Model Architecture | Input Features | Precision (%) | Recall (%) | F1-Score (%) | Accuracy (%) | ROC-AUC | Inference Latency (ms/session) | Throughput (sessions/sec) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random / Baseline** | Event Frequency | 28.40% | 51.20% | 36.52% | 62.10% | 0.5000 | **0.08 ms** | 12,500 |
| **DeepLog (Du et al.)** | Sequential Tokens | 94.73% | 92.53% | 93.62% | 94.10% | 0.9582 | 4.12 ms | 242.7 |
| **LogAnomaly (Meng et al.)** | Sequential + Quantitative | 95.97% | 92.52% | 94.22% | 95.04% | 0.9691 | 5.86 ms | 170.6 |
| **TCN (Dilated Causal)** | Sequential Tokens | 94.64% | 95.30% | 94.97% | 95.22% | 0.9745 | 1.48 ms | 675.7 |
| **Semantic Autoencoder** | Dense PCA Embeddings | **98.89%** | **95.83%** | **97.34%** | **98.78%** | **0.9912** | 0.32 ms | **3,125.0** |

```
                       F1-Score Comparison Across Architectures
    100% +-------------------------------------------------------------------+
         |                                                        [97.34%]   |
     95% |                            [94.22%]     [94.97%]          ||      |
         |              [93.62%]         ||           ||             ||      |
     90% |                 ||            ||           ||             ||      |
         |                 ||            ||           ||             ||      |
     80% |                 ||            ||           ||             ||      |
         |                 ||            ||           ||             ||      |
     50% |                 ||            ||           ||             ||      |
         |   [36.52%]      ||            ||           ||             ||      |
      0% +-------+-------------+-------------+------------+----------+-------+
              Baseline      DeepLog      LogAnomaly       TCN      Autoencoder
```

### 4.2 Key Observations
1. **False Alarm Reduction in LogAnomaly**: DeepLog suffered from 867 false positives on HDFS test sessions caused by benign asynchronous thread concurrency. LogAnomaly reduced false positives to 654 ($24.6\%$ reduction) due to its quantitative stream verifying that total event frequencies remained invariant.
2. **Recall Superiority of TCN**: TCN achieved the highest recall among next-step sequence predictors ($95.30\%$) due to its exponential receptive field ($2^0, 2^1, 2^2, \dots$), capturing long-range inter-block dependencies that vanishing gradients eroded in LSTMs.
3. **Inference Speedup**: TCN yielded a **$2.8\times$ throughput speedup** over DeepLog ($675.7$ vs $242.7$ sessions/sec). The feed-forward 1D convolutions execute concurrently across temporal dimensions without sequential state-passing dependencies.
4. **Autoencoder Accuracy & Efficiency**: The Semantic MLP Autoencoder achieved the highest F1-score ($97.34\%$) and highest throughput ($3,125$ sessions/sec). Compressing session windows into dense semantic centroids enables fast vector matrix multiplications that outperform token-by-token next-step verification.

---

## 5. Error Analysis & Domain Implications

```
+--------------------+-----------------------------------+-----------------------------------+
| Error Type         | Dominant Root Cause               | Mitigation Strategy               |
+--------------------+-----------------------------------+-----------------------------------+
| False Positive     | Benign concurrency reordering     | Multi-modal quantitative counts   |
| (FP: 654 - 908)    | Unseen rare normal maintenance    | Dynamic online template discovery |
+--------------------+-----------------------------------+-----------------------------------+
| False Negative     | Subtle silent data corruption     | Parameter-level numeric anomaly   |
| (FN: 792 - 1,259)  | Execution loop without crash      | Duration & inter-arrival tracking |
+--------------------+-----------------------------------+-----------------------------------+
```

### 5.1 False Positives (Alert Fatigue)
- **Root Cause**: Rare system operations (e.g., node decommissioning, automated disk rebalancing, software rolling upgrades) occur infrequently in normal training data. When encountered, next-key models flag these transitions as anomalous.
- **Production Impact**: Excessive false alarms lead SRE engineers to ignore alerts.
- **Solution**: Dynamic candidate set calibration ($g = 9 \to 12$) during scheduled maintenance windows, and quantitative vector regularization.

### 5.2 False Negatives (Missed Incidents)
- **Root Cause**: Silent data corruption or hanging IO threads where no explicit error log is printed, but the expected subsequent confirmation log never arrives.
- **Production Impact**: Uncaught cascading outages that escalate across data centers.
- **Solution**: Combining sequence tracking with temporal timestamp delta modeling (e.g., inter-arrival time anomalies).

---

## 6. Limitations & Future Directions

1. **Parameter Ignoring**: While Drain successfully extracts parameters into `ParameterList`, existing next-key predictors primarily classify template IDs. Future work will integrate continuous parameter modeling (e.g., monitoring write bytes, transmission latencies, and IP subnets) via multi-head attention.
2. **Concept Drift & Log LLMs**: Large distributed systems update code weekly, producing newly formatted log templates. Fine-tuning foundation models (e.g., Llama, Mistral, RoBERTa) on log syntax allows zero-shot generalization to unseen templates without parser retraining.
3. **Streaming Graph Telemetry**: Modeling multi-node interactions as spatio-temporal graphs (Graph Neural Networks) to capture causal anomaly propagation across distributed microservices.

---

## 7. Citation & Academic Attribution

If you utilize this codebase, models, or empirical benchmarks in your research, please cite:

```bibtex
@mastersthesis{jafari2023loganomaly,
  author       = {Mostafa Jafari},
  title        = {Deep Learning Based Log Anomaly Detection for Detecting Diverse Faults in Distributed Systems},
  school       = {Shahid Beheshti University},
  year         = {2023},
  address      = {Tehran, Iran},
  note         = {Faculty of Electrical and Computer Engineering}
}

@article{du2017deeplog,
  author    = {Min Du and Feifei Li and Guineng Zheng and Vivek Srikumar},
  title     = {DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning},
  journal   = {ACM SIGSAC Conference on Computer and Communications Security (CCS)},
  pages     = {1285--1298},
  year      = {2017}
}

@inproceedings{meng2019loganomaly,
  author    = {Weibin Meng and Ying Liu and Yichen Zhu and Shenglin Zhang and Dan Pei and others},
  title     = {LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {4739--4745},
  year      = {2019}
}

@inproceedings{he2017drain,
  author    = {Pinjia He and Jieming Zhu and Zibin Zheng and Michael R. Lyu},
  title     = {Drain: An Online Log Parser with Fixed Depth Tree},
  booktitle = {IEEE International Conference on Web Services (ICWS)},
  pages     = {33--40},
  year      = {2017}
}
```
