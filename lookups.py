from pathlib import Path
import json
import urllib.request
import urllib.error


BASE_DIR = Path(__file__).resolve().parent


# Gene ontology GO lookup

class GOLookup:

    def __init__(
        self,
        lookup_file="go-basic.json",
        cache_file="go_cache.json"
    ):

        self.lookup_file = Path(
            lookup_file
        )

        self.cache_file = Path(
            cache_file
        )

        self.lookup_table = (
            self.load_lookup_table()
        )

        self.cache = self.load_cache()


    def load_lookup_table(self):

        if not self.lookup_file.exists():

            print(
                f"Warning: GO lookup file not found: "
                f"{self.lookup_file}"
            )

            return {}

        try:

            with open(
                self.lookup_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            print(
                "Warning: could not read GO lookup file."
            )

            return {}


        lookup = {}

        # GO JSON stores terms in the "graphs" section
        graphs = data.get("graphs", [])

        if not graphs:
            return lookup

        graph = graphs[0]

        for node in graph.get("nodes", []):

            go_id = node.get("id")

            if not go_id:
                continue

            # Convert:
            # http://purl.obolibrary.org/obo/GO_0003677
            #
            # into:
            # GO:0003677

            if "GO_" in go_id:

                go_id = (
                    "GO:"
                    + go_id.split("GO_")[-1]
                )

            name = node.get("lbl")

            meta = node.get(
                "meta",
                {}
            )

            definition = None

            if isinstance(meta, dict):

                definition_data = meta.get(
                    "definition",
                    {}
                )

                if isinstance(
                    definition_data,
                    dict
                ):

                    definition = (
                        definition_data.get("val")
                    )

            lookup[go_id] = {
                "id": go_id,
                "name": name,
                "definition": definition
            }

        return lookup


    def load_cache(self):

        if not self.cache_file.exists():
            return {}

        try:

            with open(
                self.cache_file,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            return {}


    def save_cache(self):

        with open(
            self.cache_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.cache,
                file,
                indent=2,
                ensure_ascii=False
            )


    def lookup(self, go_id):

        go_id = str(
            go_id
        ).strip()

        if not go_id.startswith("GO:"):
            return None

        # Cache first
        if go_id in self.cache:

            return self.cache[go_id]

        # Local GO ontology
        result = self.lookup_table.get(
            go_id
        )

        if result is None:
            return None

        # Save to cache
        self.cache[go_id] = result

        self.save_cache()

        return result


    def get_description(self, go_id):

        result = self.lookup(
            go_id
        )

        if result is None:
            return None

        name = result.get(
            "name"
        )

        definition = result.get(
            "definition"
        )

        if name and definition:

            return (
                f"{name}. "
                f"{definition}"
            )

        if name:
            return name

        if definition:
            return definition

        return None    


# Enzyme Commission EC lookup

class ECLookup:

    def __init__(
        self,
        lookup_file="ec_lookup.json"
    ):

        self.lookup_file = (
            BASE_DIR / lookup_file
        )

        self.lookup_table = (
            self.load_lookup()
        )

    def load_lookup(self):

        if not self.lookup_file.exists():

            print(
                f"Warning: EC lookup file not found: "
                f"{self.lookup_file}"
            )

            return {}

        try:

            with open(
                self.lookup_file,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            print(
                "Warning: could not read EC lookup."
            )

            return {}

    def lookup(self, ec_id):

        ec_id = str(
            ec_id
        ).strip()

        return self.lookup_table.get(
            ec_id
        )

    def get_description(self, ec_id):

        result = self.lookup(
            ec_id
        )

        if result is None:
            return None

        if isinstance(
            result,
            str
        ):

            return result

        if isinstance(
            result,
            dict
        ):

            return (
                result.get("name")
                or result.get("definition")
            )

        return None



# KEGG lookup

class KEGGLookup:

    def __init__(
        self,
        cache_file="kegg_cache.json"
    ):

        self.cache_file = (
            BASE_DIR / cache_file
        )

        self.cache = self.load_cache()


    def load_cache(self):

        if not self.cache_file.exists():

            print(
                f"Warning: KEGG cache not found: "
                f"{self.cache_file}"
            )

            return {}

        try:

            with open(
                self.cache_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            print(
                "Warning: could not read KEGG cache."
            )

            return {}

        if not isinstance(data, dict):

            print(
                "Warning: KEGG cache does not contain "
                "a valid dictionary."
            )

            return {}

        print(
            f"KEGG cache loaded: "
            f"{len(data)} entries."
        )

        return data


    def lookup(self, kegg_id):

        kegg_id = str(
            kegg_id
        ).strip()

        if not kegg_id.startswith("K"):
            return None

        # --------------------------------------------------
        # LOCAL CACHE ONLY
        # --------------------------------------------------

        return self.cache.get(
            kegg_id
        )


    def get_description(self, kegg_id):

        result = self.lookup(
            kegg_id
        )

        if result is None:
            return None

        # Cache entries may be stored as dictionaries
        # or directly as strings.

        if isinstance(
            result,
            str
        ):

            return result

        if isinstance(
            result,
            dict
        ):

            return result.get(
                "name"
            ) or result.get(
                "definition"
            )

        return None


#testing

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TESTING LOOKUP DATABASES")
    print("=" * 60)

    # GO
    go = GOLookup()

    print()
    print("GO TEST")

    go_id = "GO:0003677"

    print(
        "ID:",
        go_id
    )

    print(
        "Description:",
        go.get_description(go_id)
    )

    # EC
    ec = ECLookup()

    print()
    print("EC TEST")

    ec_id = "2.7.7.7"

    print(
        "ID:",
        ec_id
    )

    print(
        "Description:",
        ec.get_description(ec_id)
    )

    # KEGG
    kegg = KEGGLookup()

    print()
    print("KEGG TEST")

    kegg_id = "K02313"

    print(
        "ID:",
        kegg_id
    )

    print(
        "Description:",
        kegg.get_description(kegg_id)
    )
