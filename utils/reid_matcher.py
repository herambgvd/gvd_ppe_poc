"""
# ======================================
# REID MATCHER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Stores worker appearance embeddings.
- Matches workers using cosine similarity.
- Uses temporal embedding memory.
- Prevents unstable identity fragmentation.
- Creates enterprise-level persistent identities.
"""

from __future__ import annotations

import uuid
from collections import deque

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ======================================
# REID MATCHER
# ======================================

class ReIDMatcher:

    # ======================================
    # INIT
    # ======================================

    def __init__(self):

        # ======================================
        # IDENTITY DATABASE
        # ======================================

        # Stores averaged embeddings
        self.identity_database = {}

        # ======================================
        # TEMPORAL EMBEDDING MEMORY
        # ======================================

        # {
        #   gid_xxx: deque([...])
        # }

        self.embedding_history = {}

        # ======================================
        # MATCH THRESHOLD
        # ======================================

        self.threshold = 0.60

        # ======================================
        # UNKNOWN FRAME COUNTER
        # ======================================

        self.unknown_counter = 0

        # ======================================
        # UNKNOWN LIMIT
        # ======================================

        # Number of failed matches required
        # before creating new identity.

        self.max_unknown_frames = 5

        print("[ReID] Matcher initialized")

    # ======================================
    # MATCH EMBEDDING
    # ======================================

    def match(self, embedding):

        # ======================================
        # INVALID INPUT
        # ======================================

        if embedding is None:
            return None

        # ======================================
        # FORCE NUMPY ARRAY
        # ======================================

        try:

            if not isinstance(
                embedding,
                np.ndarray
            ):

                embedding = np.array(
                    embedding
                )

            embedding = embedding.flatten()

        except Exception as e:

            print(
                f"[ReID ERROR] {e}"
            )

            return None

        # ======================================
        # FIRST IDENTITY
        # ======================================

        if len(self.identity_database) == 0:

            new_identity = (
                self._create_identity(
                    embedding
                )
            )

            return {

                "identity_id": new_identity,

                "similarity": 1.0,

                "is_new": True
            }

        # ======================================
        # FIND BEST MATCH
        # ======================================

        best_identity = None

        best_similarity = -1

        # ======================================
        # COMPARE AGAINST MEMORY
        # ======================================

        for identity_id, history in (

            self.embedding_history.items()

        ):

            try:

                # ======================================
                # TEMPORAL AVERAGE
                # ======================================

                stored_embedding = np.mean(

                    history,

                    axis=0
                )

                # ======================================
                # COSINE SIMILARITY
                # ======================================

                similarity = cosine_similarity(

                    [embedding],

                    [stored_embedding]

                )[0][0]

                if similarity > best_similarity:

                    best_similarity = similarity

                    best_identity = identity_id

            except Exception as e:

                print(
                    f"[ReID Similarity ERROR] {e}"
                )

        # ======================================
        # MATCH FOUND
        # ======================================

        if (

            best_identity is not None

            and

            best_similarity >= self.threshold

        ):

            # ======================================
            # RESET UNKNOWN COUNTER
            # ======================================

            self.unknown_counter = 0

            # ======================================
            # UPDATE TEMPORAL MEMORY
            # ======================================

            self.embedding_history[
                best_identity
            ].append(embedding)

            # ======================================
            # UPDATE AVERAGED EMBEDDING
            # ======================================

            self.identity_database[
                best_identity
            ] = np.mean(

                self.embedding_history[
                    best_identity
                ],

                axis=0
            )

            return {

                "identity_id": best_identity,

                "similarity": float(best_similarity),

                "is_new": False
            }

        # ======================================
        # TEMPORARY UNKNOWN
        # ======================================

        self.unknown_counter += 1

        print(

            f"[ReID] Unknown Counter: "

            f"{self.unknown_counter}"

        )

        # ======================================
        # WAIT BEFORE NEW IDENTITY
        # ======================================

        if (

            self.unknown_counter

            <

            self.max_unknown_frames

        ):

            return {

                "identity_id": (

                    best_identity

                    if best_identity is not None

                    else "temporary_unknown"
                ),

                "similarity": float(best_similarity),

                "is_new": False
            }

        # ======================================
        # RESET COUNTER
        # ======================================

        self.unknown_counter = 0

        # ======================================
        # CREATE NEW IDENTITY
        # ======================================

        new_identity = (
            self._create_identity(
                embedding
            )
        )

        return {

            "identity_id": new_identity,

            "similarity": float(best_similarity),

            "is_new": True
        }

    # ======================================
    # CREATE NEW IDENTITY
    # ======================================

    def _create_identity(self, embedding):

        new_identity = (

            "gid_"

            +

            uuid.uuid4().hex[:12]
        )

        # ======================================
        # TEMPORAL MEMORY
        # ======================================

        self.embedding_history[
            new_identity
        ] = deque(maxlen=50)

        self.embedding_history[
            new_identity
        ].append(embedding)

        # ======================================
        # INITIAL EMBEDDING
        # ======================================

        self.identity_database[
            new_identity
        ] = embedding

        print(
            f"[ReID] New Identity: "
            f"{new_identity}"
        )

        return new_identity

    # ======================================
    # TOTAL IDENTITIES
    # ======================================

    def total_identities(self):

        return len(
            self.identity_database
        )

    # ======================================
    # RESET DATABASE
    # ======================================

    def reset(self):

        self.identity_database.clear()

        self.embedding_history.clear()

        self.unknown_counter = 0

        print(
            "[ReID] Identity database cleared"
        )