from pathlib import Path

import streamlit as st

import lib.config as config
import lib.langchain_impl as langchain_impl
import lib.nav as nav

st.set_page_config(
    page_title="OuRAGboros",
    page_icon=":snake:",
    layout="wide",
)
nav.pages()

st.title(":snake: LangChain OuRAGboros")

# Initialize session state
#
if "documents" not in st.session_state:
    st.session_state.documents = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""


def perform_document_retrieval(
        query: str,
        k=3,
        score_threshold: float = 0.2,
        model: str = config.default_model,
        use_opensearch_vectorstore: bool = False,
):
    """
    Retrieves a list of Document objects that correspond to the provided search query.
    :param query:
    :param k:
    :param score_threshold:
    :param model:
    :param use_opensearch_vectorstore:
    :return:
    """
    if use_opensearch_vectorstore:
        import langchain_community.vectorstores.opensearch_vector_search as os_vs

        st.text("Ensuring OpenSearch index existence...")
        langchain_impl.ensure_opensearch_index(model)

        return langchain_impl.opensearch_doc_vector_store(
            embedding_model).similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
            search_type=os_vs.SCRIPT_SCORING_SEARCH,
            # See: https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/#spaces
            #
            space_type="cosinesimil"
        )
    else:
        # Uses cosine similarity by default.
        # See: https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html#langchain_core.vectorstores.in_memory.InMemoryVectorStore
        #
        vector_store = langchain_impl.get_in_memory_vector_store(model)
        # We add 1 to the score to keep formatting consistent with OpenSearch
        return [
            (d, s + 1) for (d, s) in vector_store.similarity_search_with_score(query, k=k)
            if s + 1 >= score_threshold
        ]


with st.sidebar:
    st.header("Search Configuration")
    use_opensearch = st.toggle(
        "Use OpenSearch",
        config.prefer_opensearch,
        help=f"Requires an OpenSearch instance running at {config.opensearch_base_url}. If "
             f"this toggle is off, all documents are retrieved from an in-memory "
             f"vector store which is lost when the application terminates.",
    )
    embedding_model = st.selectbox(
        "Embedding model:",
        langchain_impl.get_available_embeddings(),
        index=0,
    )
    llm_model = st.selectbox(
        "LLM:",
        langchain_impl.get_available_llms(),
        index=0,
    )
    query_result_score_inf = st.slider(
        "Set the document match score threshold:",
        value=1.0,
        min_value=0.0,
        max_value=2.0,
        help="Score is computed using cosine similarity plus 1 to ensure a non-negative "
             "score."
    )
    return_documents = st.slider(
        "Set the maximum retrieved documents:",
        value=3,
        min_value=1,
        max_value=15,
        step=1,
        help="Sets an upper bound on the number of documents to return."
    )
    llm_prompt = st.text_area("Model Prompt", config.default_prompt, height=250)

search_query = st.chat_input("Enter a search query")


@st.fragment
def _render_source_docs(docs):
    """
    Renders source documents used to generate LLM context.

    We put this function inside its own fragment so that download links don"t trigger a
    full page refresh.

    :param docs:
    :return:
    """
    for i, (doc, score) in enumerate(docs):
        if i:
            st.divider()
        st.subheader(
            "{}_page{}_chunk{}_overlap{}".format(
                Path(doc.metadata["source"]).name,
                doc.metadata["page_number"],
                doc.metadata["chunk_index"],
                doc.metadata["chunk_overlap_percent"],
            )
        )
        st.download_button(
            label="Download as text",
            data=doc.page_content,
            file_name=Path(doc.metadata["source"]).name,
            mime="text/plain",
            key=Path(doc.id)

        )
        st.markdown(f"**Score:** {score}")
        st.markdown('**Document Text:**')
        st.code('{}{}'.format(
            doc.page_content[:500],
            '... [download file to see more]' if len(doc.page_content) > 100 else ''
        ),
            language=None,
            line_numbers=True,
            wrap_lines=True,
        )


# Save search query if a new one was provided
#
if search_query:
    st.session_state.search_query = search_query

# Main page content
#
if st.session_state.search_query and llm_model:
    with st.chat_message("user"):
        st.text(st.session_state.search_query)

    with st.spinner(f"Loading `{embedding_model}` embeddings..."):
        langchain_impl.pull_model(embedding_model)

    with st.spinner(f"Loading `{llm_model}` LLM..."):
        langchain_impl.pull_model(llm_model)

    with st.chat_message("ai"):
        st.text("Searching knowledge base for relevant documentation...")

        matches = perform_document_retrieval(
            st.session_state.search_query,
            k=return_documents,
            score_threshold=query_result_score_inf,
            model=embedding_model,
            use_opensearch_vectorstore=use_opensearch,
        )
        st.session_state.documents = matches

        singular_match = len(st.session_state.documents) == 1

        if len(st.session_state.documents):
            st.text("Found {} document match{}.".format(
                len(st.session_state.documents),
                "" if singular_match else "es"
            ))
        else:
            st.warning(
                "No document matches found. Try a new query or lower the score "
                "threshold for a more context-aware response."
            )

        context = "\n".join([
            doc.page_content for doc, score in st.session_state.documents
        ])

        st.session_state.llm_response = langchain_impl.ask_llm(
            llm_model,
            llm_prompt,
            st.session_state.search_query,
            context=context,
        )
        st.write_stream(st.session_state.llm_response)

        if len(st.session_state.documents):
            with st.expander(f"Source document{"" if singular_match else "s"}"):
                _render_source_docs(st.session_state.documents)
