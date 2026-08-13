#get testing data that would be what blast produces in the real program
#use python library- machine learning to analyze it
#consensus
#generate a description of what the protein is doing- function
#product description (english name)
#confidence- how good e values are- same function

from pathlib import Path
from collections import defaultdict, Counter

import re
import math
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from lookups.lookups import GOLookup, ECLookup, KEGGLookup

class AnnotationEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

        self.go_lookup = GOLookup()
        self.ec_lookup = ECLookup()
        self.kegg_lookup = KEGGLookup()

        # Raw input table

        self.data = None

        # Query grouped information

        self.grouped_hits = {}

        # AI pipeline

        self.embeddings = {}
        self.similarity = {}
        self.clusters = {}

        # Final annotation results

        self.annotations = {}
        self.lookup_cache = {}
        self.results = {}


    def read_hits(self, filename: str):
        """
        Reads BLAST/DIAMOND annotation results
        format:query   hit   annotation   E-value
        may contain multiple GO/EC/KEGG identifiers w spaces
        """

        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(filename)

        records = []

        with open(filename, "r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip header
                if line.startswith("#"):
                    continue

                # Split on any amount of whitespace
                parts = line.split()

                #at least: query + hit + annotation + e-value
                if len(parts) < 4:
                    print(
                        f"Warning: skipping broen line "
                        f"{line_number}: {line}"
                    )
                    continue

                #spli by position
                query = parts[0]
                hit = parts[1]
                evalue = float(parts[-1])
                annotation = " ".join(parts[2:-1])

                records.append({
                    "query": query,
                    "hit": hit,
                    "annotation": annotation,
                    "evalue": evalue
                })

        if not records:
            raise ValueError("No valid annotation records were found in the input file.")

        self.data = pd.DataFrame(records)
        return self.data


    def load_lookup_tables(self):
        """
        Initialize GO/EC/KEGG lookup databases.
        classes in lookups.py-handle own JSON files,caches, and API/database access.
        """

        print("Loading annotation lookup databases...")

        # Lookup objects are already initialized in __init__
        # just to verify if available.

        print("GO lookup ready.")
        print("EC lookup ready.")
        print("KEGG lookup ready.")


    ''''def clean_description(self, text):
        text = text.lower()
        text = re.sub(r"\(.*?\)","",text        )

        remove_words = [
            "putative",
            "probable",
            "predicted",
            "hypothetical",
            "possible",
            "protein"]

        for word in remove_words:
            text = text.replace(word, "")

        text = re.sub(r"\s+", " ", text)
        return text.strip()'''


    def classify_annotation(self, annotation):
        #which identifier
        #Priority: EC > GO > KEGG ---- EC is very enzyme centered- 
        # most official way for enzymes. GO has all kinds of things- not just enzymes. 
        # KEGG sometimes also hold EC- K is similar to GO- uses many things- more specific. 
        # Roundabouts when not that directed to enzymes. EC is the standard for enzymes. 
        # GO is more common/frequent than KEGG, which is just KEGG. 

        annotation = str(annotation)

        if re.search(r"\b\d+\.\d+\.\d+\.\d+\b", annotation):
            return "EC"

        if re.search(r"\bGO:\d+\b", annotation):
            return "GO"

        if re.search(r"\bK\d{5}\b",annotation):
            return "KEGG"

        return "UNKNOWN"
    

    def split_annotations(self, annotation):
        #if more than 1 annotation identifier
        return str(annotation).split()


    def lookup_annotation(self, annotation):
       #Convert biological identifiers into English descriptions.
       #GO, EC,KEGG lookups are handled by the corresponding lookup classes in lookups.py
        
        annotation = str(annotation).strip()

        # EC
        ec_ids = re.findall(r"\b\d+\.\d+\.\d+\.\d+\b", annotation)

        # GO
        go_ids = re.findall( r"\bGO:\d+\b", annotation)

        # KEGG
        kegg_ids = re.findall(r"\bK\d{5}\b", annotation)

        descriptions = []

        # EC
        for identifier in ec_ids:
            description = self.ec_lookup.get_description(identifier)
            if description:
                descriptions.append(description)

        # GO
        for identifier in go_ids:
            description = self.go_lookup.get_description(identifier)
            if description:
                descriptions.append(description)

        # KEGG
        for identifier in kegg_ids:
            description = self.kegg_lookup.get_description(identifier)
            if description:
                descriptions.append(description)

        # Remove duplicates
        descriptions = list(dict.fromkeys(descriptions))

        if not descriptions:
            return "Unknown protein function"

        return "; ".join(descriptions)


    def lookup_identifier(self, identifier):
        #Looks up a single biological annotation identifier and returns its English descriptio
        annotation_type = self.classify_annotation(identifier)

        if annotation_type == "GO":
            return self.lookup_go(identifier)

        elif annotation_type == "EC":
            return self.lookup_ec(identifier)

        elif annotation_type == "KEGG":
            return self.lookup_kegg(identifier)

        return None

    
    def group_hits(self):
        """
        group all homolog hits belonging to the same query
        Every hit stores:
            -database match
            -annotation identifier(s)
            -annotation type
            -eglish biological description
            -evalue"""

        grouped = defaultdict(list)

        for _, row in self.data.iterrows():
            annotation = str(row["annotation"]).strip()
            hit_value = row["hit"]
            annotation_type = (self.classify_annotation(annotation))
            description = (self.lookup_annotation(annotation))

            grouped[row["query"]].append({
                "match": hit_value,
                "annotation": annotation,
                "type": annotation_type,
                "description": description,
                "evalue": float(row["evalue"])})

        self.grouped_hits = dict(grouped)
        return self.grouped_hits


    def summary(self):

        print()
        print("Queries:", len(self.grouped_hits))
        print("Total hits:", len(self.data))
        print()

        sizes = [len(v) for v in self.grouped_hits.values()]

        print(f"Average hits/query: {sum(sizes)/len(sizes):.1f}")


    def generate_embeddings(self):
        #semantic embeddings for every translated biological description.

        print("Generating semantic embeddings...")
        embeddings = {}

        for query, hits in self.grouped_hits.items():
            descriptions = [hit["description"] for hit in hits]

            vectors = self.model.encode(
                descriptions,
                convert_to_numpy=True,
                normalize_embeddings=True)

            embeddings[query] = {
                "hits": hits,
                "descriptions": descriptions,
                "vectors": vectors}

        self.embeddings = embeddings

        print(
            f"Generated embeddings for "
            f"{len(embeddings)} queries.")

        return embeddings


    def compute_similarity(self):
        #pairwise cosine similarity between all descriptions in to one query
  
        similarity = {}

        for query, data in self.embeddings.items():
            matrix = cosine_similarity(data["vectors"])
            similarity[query] = matrix

        self.similarity = similarity
        return similarity


#these values?
    def cluster_descriptions(self, eps=0.25, min_samples=2):
        #cluster semantically similar protein descriptions w DBSCAN and cosine distance

        print("Clustering descriptions with DBSCAN...")
        clusters = {}

        for query, data in self.embeddings.items():
            vectors = data["vectors"]
            if len(vectors) < min_samples:
                clusters[query] = np.full(len(vectors), -1)
                continue

            model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="cosine")

            labels = model.fit_predict(vectors)
            clusters[query] = labels

        self.clusters = clusters
        return clusters


    def annotation_type(self, annotation):
        #type of biological annotation. Priority:EC > GO > KEGG

        annotation = str(annotation).strip()

        # EC number
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", annotation):
            return "EC"

        # GO term
        if "GO:" in annotation:
            return "GO"

        # KEGG KO
        if re.search(r"\bK\d{5}\b", annotation):
            return "KEGG"

        return "OTHER"


    def evalue_weight(self, evalue):
        # convert an E-value into a numerical confidence weight (smaller E-values = stronger sequence similarity)

        evalue = float(evalue)

        if evalue <= 0:
            return 1.0

        # -log10(E-value), capped to avoid extreme values
        score = -np.log10(evalue)
        return min(score / 200.0, 1.0)
    

    def consensus_annotation(self):
        """
        Consensus for each query determined w:

        1. Dominant DBSCAN cluster
        2. Annotation type priority: EC > GO > KEGG
        3. Frequency of individual annotation identifiers
        4. E-value quality
        5. Semantic cluster support

        then converted to english with lookup 
        """

        results = {}

        # Annotation type priority
        type_priority = {
            "EC": 3,
            "GO": 2,
            "KEGG": 1,
            "OTHER": 0}

        # Process each query independently
        for query in self.grouped_hits:
            hits = self.grouped_hits[query]
            labels = self.clusters[query]

            # dominant DBSCAN cluster
            valid = labels != -1

            if np.any(valid):

                unique, counts = np.unique(labels[valid], return_counts=True)
                dominant = unique[np.argmax(counts)]
                cluster_indices = np.where(labels == dominant)[0]
            else:
                # No useful DBSCAN cluster:use every hit as fallback
                cluster_indices = np.arange(len(hits))

            # Store evidence for each individual annotation
            annotation_scores = defaultdict(float)
            annotation_support = defaultdict(int)
            annotation_evalues = defaultdict(list)
            annotation_types = {}

            # check every hit in dominant cluster

            for index in cluster_indices:
                hit = hits[index]
                raw_annotation = str(hit["annotation"]).strip()

                if not raw_annotation:
                    continue

                evalue = float(hit["evalue"])

                # Split annotation into individual identifiers (ex, when there are many lined up GO values)

                annotation_ids = re.findall(r"(GO:\d+|K\d+|\d+\.\d+\.\d+\.\d+)", raw_annotation)

                # If nothing matches the known formats, treat entire annotation as one
                if not annotation_ids:
                    annotation_ids = [raw_annotation]

                # E-value contribution
                evalue_score = self.evalue_weight(evalue)

                # count every individual annotation
                for annotation in annotation_ids:
                    annotation = annotation.strip()
                    if not annotation:
                        continue

                    annotation_type = self.annotation_type(annotation)
                    annotation_types[annotation] = annotation_type

                    # priority multiplier (EC > GO > KEGG)

                    priority_multiplier = {
                        "EC": 1.5,
                        "GO": 1.0,
                        "KEGG": 0.75,
                        "OTHER": 0.5
                    }.get(annotation_type, 0.5)

                    # Scoring:
                    # support matters most, e-value strengthens -- type preference.
                    contribution = (priority_multiplier * (0.5 + evalue_score))

                    annotation_scores[annotation] += contribution

                    annotation_support[annotation] += 1

                    annotation_evalues[annotation].append(evalue)

            # If no usable annotation
            if not annotation_scores:
                results[query] = {
                    "annotation": None,
                    "annotation_type": None,
                    "description": None,
                    "confidence": 0.0,
                    "support": 0,
                    "cluster_size": len(cluster_indices),
                    "total_hits": len(hits)}
                continue


            # Select consensus annotation: primary criterion is the evidence score.
            # If scores are extremely close, annotation
            # priority and support act as tie-breakers.

            best_annotation = max(annotation_scores, key=annotation_scores.get)
            best_type = annotation_types[best_annotation]
            description = self.lookup_annotation(best_annotation)
            support = annotation_support[best_annotation]
            evalues = annotation_evalues[best_annotation]

            # Best E-value supporting consensus
            best_evalue = min(evalues)

            # Semantic coherence

            cluster_vectors = self.embeddings[query]["vectors"][cluster_indices]
            centroid = cluster_vectors.mean(axis=0)
            coherence_scores = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()

            semantic_coherence = float(np.mean(coherence_scores))

            # Consensus confidence
            total_score = sum(annotation_scores.values())

            consensus_fraction = (annotation_scores[best_annotation]/ total_score
                if total_score > 0
                else 0)

            support_fraction = (support / len(cluster_indices)
                if len(cluster_indices) > 0
                else 0)

            evalue_confidence = np.mean([self.evalue_weight(e)
                for e in evalues])

            #
            confidence = (
                0.40 * consensus_fraction
                + 0.25 * support_fraction
                + 0.35 * evalue_confidence)

            # save result

            results[query] = {
                "annotation": best_annotation,
                "annotation_type": best_type,
                "description": description,
                "confidence": round(confidence, 3),
                "support": support,
                "cluster_size": len(cluster_indices),
                "total_hits": len(hits),
                "mean_evalue": float(np.mean(evalues)),
                "best_evalue": best_evalue,
                "semantic_coherence": round(semantic_coherence, 3),
                "score": round(annotation_scores[best_annotation],3)}

        self.annotations = results
        return results



    def evalue_to_score(self, evalue, max_score=200):
        #onverts E-value into  normalized confidence score.
        #format: decimal between 0 (least) and 1 (most confidence)

        try:
            e = float(evalue)
        except Exception:
            return 0.0

        # DIAMOND prints 0.0 for extremely significant hits
        if e == 0:
            return 1.0

        score = -math.log10(e)
        score = min(score, max_score)
        return score / max_score


    def generate_product_description(self):

        for query, result in self.annotations.items():
            description = result.get("description")
            annotation_type = result.get("annotation_type")

            if not description:
                result["product"] = ("Uncharacterized protein")
                continue

            # uncertainty
            description = re.sub(r"\bputative\b", "", description, flags=re.IGNORECASE)
            description = re.sub(r"\bprobable\b", "", description, flags=re.IGNORECASE)
            description = re.sub(r"\bpredicted\b","", description, flags=re.IGNORECASE)
            description = re.sub(r"\bhypothetical\b","", description, flags=re.IGNORECASE)
            description = re.sub(r"\s+", " ", description).strip()
            result["product"] = description
        return self.annotations


    def print_results(self, limit=20):
        print()
        print("=" * 80)
        print("ANNOTATION RESULTS")
        print("=" * 80)

        for i, (query, result) in enumerate(self.annotations.items()):
            if i >= limit:
                break
            print()
            print("Query:", query)
            print("Consensus:", result.get("annotation"))
            print("Product:", result.get("product"))
            print("Type:", result.get("annotation_type"))
            print("Confidence:", result.get("confidence"))
            print("Support:", f"{result.get('support')}/"
                f"{result.get('total_hits')}")
            print("Semantic coherence:",
                result.get("semantic_coherence", "N/A"))
            print("Best E-value:",
                result.get("best_evalue", "N/A"))



if __name__ == "__main__":
    engine = AnnotationEngine()

    # load annotation databases
    engine.load_lookup_tables()

    # read BLAST/DIAMOND results
    engine.read_hits("hits.tsv")

    # group homologs by query
    engine.group_hits()
    engine.summary()

    # convert annotations to semantic representations
    engine.generate_embeddings()

    # calculate semantic similarity
    engine.compute_similarity()

    # cluster annotations
    engine.cluster_descriptions(eps=0.25, min_samples=2)

    # determine consensus
    engine.consensus_annotation()

    # generate the english descs
    engine.generate_product_description()

    # display results
    engine.print_results(limit=20)


#More notes: GO appears most but ec good - find for most common  - k are kegg numbers lower priorities. ec>go>kegg
