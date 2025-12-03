import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
import nest_asyncio
nest_asyncio.apply()

# -----------------------------
# TruLens latest version imports
# -----------------------------
from trulens.core import Feedback, Select
from trulens.providers.openai import OpenAI as TruOpenAI
from trulens.apps.llamaindex import TruLlama

os.environ["TRULENS_OTEL_TRACING"] = "0"

# ------------------------------------
# Load API keys
# ------------------------------------
def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")

def get_hf_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("HUGGINGFACE_API_KEY")


# TruLens LLM Provider
openai_provider = TruOpenAI(api_key=get_openai_api_key())

# ------------------------------------
# FEEDBACK FUNCTIONS
# ------------------------------------

# QA relevance
qa_relevance = (
    Feedback(openai_provider.relevance, name="Answer Relevance")
        .on_input_output()
)

# Query-source context relevance - using select_context instead of select_source_nodes
qs_relevance = (
    Feedback(openai_provider.relevance, name="Context Relevance")
        .on_input()
        .on(TruLlama.select_context())
        .aggregate(np.mean)
)


# Groundedness - using select_context
groundedness = (
    Feedback(openai_provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_context())
        .on_output()
)

# Register feedbacks
feedbacks = [qa_relevance, qs_relevance, groundedness]

# ------------------------------------
# TruLlama Recorder Helpers
# ------------------------------------
def get_trulens_recorder(query_engine, feedbacks, app_id):
    """
    Create a TruLlama recorder with custom feedbacks.
    Note: Pass query_engine without calling it first (no parentheses).
    """
    return TruLlama(
        query_engine,
        app_name=app_id,
        feedbacks=feedbacks
    )

def get_prebuilt_trulens_recorder(query_engine, app_id):
    """
    Create a TruLlama recorder with prebuilt feedbacks.
    Note: Pass query_engine without calling it first (no parentheses).
    """
    return TruLlama(
        query_engine,
        app_name=app_id,
        feedbacks=feedbacks
    )

# ------------------------------------
# LlamaIndex code (updated for latest version)
# ------------------------------------
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


def build_sentence_window_index(
    document, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents([document])
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return sentence_index


def get_sentence_window_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")

    return sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank]
    )


# ------------------------------------
# AutoMerging Index Code
# ------------------------------------
def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever,
        automerging_index.storage_context,
        verbose=True
    )

    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base"
    )

    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[rerank]
    )
