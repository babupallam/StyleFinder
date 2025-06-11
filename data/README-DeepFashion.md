## Fashion Retrieval Benchmark Analysis

---
# dataset found

http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html




### 1. DeepFashion Benchmark Analysis

#### 1.1 Category and Attribute Prediction Benchmark

* **Size**: 289,222 images
* **Annotations**: 50 categories (e.g., dress, shirt, pants), 1,000 attributes (e.g., floral, short sleeve, cotton), bounding boxes
* **Pros**: Great for generating text prompts (e.g., "a sleeveless floral dress"), fine-tuning CLIP for attribute detection, fully public, clean centered images
* **Cons**: No retrieval task or similarity matching, lacks query-gallery split
* **Verdict**: Useful for prompt training and fine-tuning, but requires additional splits for retrieval evaluation

#### 1.2 In-shop Clothes Retrieval Benchmark

* **Size**: 52,712 images
* **Identities**: 7,982 clothing items
* **Splits**: Pre-defined query and gallery sets showing the same clothes in different views/poses
* **Pros**: Perfect for imageimage retrieval, defined evaluation metrics (Top-K accuracy)
* **Cons**: Access requires agreement, limited background variation
* **Verdict**: Highly recommended as the primary retrieval benchmark

#### 1.3 Consumer-to-shop Clothes Retrieval Benchmark

* **Size**: 239,557 images
* **Pairs**: Consumer (real-world) vs. shop (product) images with domain gap
* **Pros**: Realistic queries, strong contrastive pairs, more challenging
* **Cons**: Requires access, stricter evaluation
* **Verdict**: Ideal for advanced generalization testing after In-shop model

#### 1.4 Fashion Landmark Detection Benchmark

* **Size**: 123,016 images
* **Annotations**: Body landmarks (neck, sleeve, hem, waist)
* **Pros**: Fine-grained pose/structure analysis
* **Cons**: Not retrieval-focused
* **Verdict**: Not needed for visual similarity

#### 1.5 DeepFashion-MultiModal (2022)

* **Contents**: Images plus text descriptions, parsing masks, keypoints, fine-grained labels
* **Pros**: Multi-modal learning, segmentation and pose integration
* **Cons**: Access approval needed, complex format
* **Verdict**: Useful for future multi-modal extensions

---

### 2. In-shop Clothes Retrieval Benchmark Deep Dive

* **Database Link**: [https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc?resourcekey=0-NWldFxSChFuCpK4nzAIGsg](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc?resourcekey=0-NWldFxSChFuCpK4nzAIGsg) fileciteturn0file1
* **Structure**:

  * Query set: one image per identity used to search
  * Gallery set: remaining images per identity
* **Evaluation Metrics**: Top-K accuracy (Rank-1, Rank-5), CMC curve, mAP

---

### 3. Final Recommendations

| Benchmark                       | Best Use                       | Fit for Project                             |
| ------------------------------- | ------------------------------ | ------------------------------------------- |
| In-shop Clothes Retrieval       | ImageImage retrieval          | **Primary choice**                          |
| Consumer-to-shop Retrieval      | Real-world robustness          | Secondary stage for generalization          |
| Category & Attribute Prediction | Prompt generation, fine-tuning | Good for CLIP attribute tuning              |
| Fashion Landmark Detection      | Pose/structure analysis        | Not relevant for retrieval                  |
| DeepFashion-MultiModal          | Multi-modal learning           | Future extension (text, parse, pose fusion) |

