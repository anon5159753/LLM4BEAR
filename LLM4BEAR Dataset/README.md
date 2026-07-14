## LLM4BEAR Dataset

Click the individual folders **Electronic**, **Clothing**, and **Food** to access the **LLM4BEAR-refined datasets**. In addition to high-quality bundles with labelled intents, we have provided LLM-enriched item metadata that may be useful for downstream bundle generation and recommendation tasks.

### a. Dataset Statistics

| Statistic | Electronic | Clothing | Food |
| :--- | :---: | :---: | :---: |
| **#Items** | 3499 | 4487 | 3767 |
| **#Sessions** | 893 | 968 | 914 |
| **#Bundles** | 1750 | 1910 | 1784 |
| **#Intents** | 1750 | 1910 | 1784 |
| **Average Bundle Size** | 2.80 | 2.66 | 3.23 |
---
**Note**: There are less sessions in this dataset, as in the **BundleRec dataset**, not all sessions necessarily had a valid bundle (i.e., Bundle size $\geq 2$ or simply cleaned). Thus, the number of sessions denoted is the number of usable sessions for downstream bundle generation experiments.

### b. Description of Dataset Files

| File Name | Description |
| :--- | :--- |
| **bundle_intent.csv** | This file contains bundles and their annotated intents. This is a tab separated list with 2 columns: `bundle ID` \| `intent` \| |
| **bundle_item.csv** | This file contains bundles and their associated items. Each bundle has at least 2 items. This is a tab separated list with 2 columns: `bundle ID` \| `item ID` \| |
| **session_item.csv** | This file contains sessions and their associated items. Each session has at least 2 items. This is a tab separated list with 2 columns: `session ID` \| `item ID` \| |
| **items_enriched.csv** | This file contains the LLM-enriched meta of each item in the dataset. This is a tab separated list with 10 columns: `item ID` \| `titles` \| `description` \| `product_type` \| `brand` \| `design_focus/Dietary_considerations` \| `target_user/flavor_profile` \| `cost_tier` \| `key_features` \| `categories` \| |

Other files are unchanged from **BundleRec**: [https://github.com/BundleRec/bundle_recommendation](https://github.com/BundleRec/bundle_recommendation)

.pkl files provided for quick easy access of bundle intents and items.
