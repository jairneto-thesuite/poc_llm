import os
import streamlit as st

# Import Haystack and Apify integration components
from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from apify_haystack import ApifyDatasetFromActorCall

# ------------------------------
# Utility Functions & Pipeline
# ------------------------------

# Improved dataset mapping function (with content truncation)
def dataset_mapping_function(dataset_item: dict) -> Document:
    max_chars = 10000
    content = dataset_item.get("markdown", "")
    return Document(
        content=content[:max_chars],
        meta={
            "title": dataset_item.get("metadata", {}).get("title"),
            "url": dataset_item.get("metadata", {}).get("url"),
            "language": dataset_item.get("metadata", {}).get("languageCode")
        }
    )

# Create the Haystack pipeline
def create_pipeline(query: str) -> Pipeline:
    document_loader = ApifyDatasetFromActorCall(
        actor_id="apify/rag-web-browser",
        run_input={
            "query": query,
            "maxResults": 2,
            "outputFormats": ["markdown"]
        },
        dataset_mapping_function=dataset_mapping_function,
    )
    
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True
    )
    
    prompt_template = """
    Analyze the following content and provide:
    1. Key points and findings
    2. Practical implications
    3. Notable conclusions
    Be concise.
    
    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    
    Analysis:
    """
    
    prompt_builder = PromptBuilder(template=prompt_template)
    generator = OpenAIGenerator(model="gpt-4o-mini")
    
    pipe = Pipeline()
    pipe.add_component("loader", document_loader)
    pipe.add_component("cleaner", cleaner)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)
    
    pipe.connect("loader", "cleaner")
    pipe.connect("cleaner", "prompt_builder")
    pipe.connect("prompt_builder", "generator")
    
    return pipe

# Function to run the pipeline and return the analysis result
def research_topic(query: str) -> str:
    pipeline = create_pipeline(query)
    result = pipeline.run(data={})
    # Assuming the first reply is the desired output
    return result["generator"]["replies"][0]

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("The Suite PoC")

st.markdown("Get instant insights from the latest web content on any topic. "
           "Simply enter your API tokens and type in what you'd like to learn about, "
           "then click **Run Analysis** to receive a clear, organized summary.")

# API Token inputs (entered as passwords)
apify_token = st.text_input("Enter YOUR APIFY_API_TOKEN", type="password")
openai_token = st.text_input("Enter YOUR OPENAI_API_KEY", type="password")

# Set API keys as environment variables if provided
if apify_token:
    os.environ["APIFY_API_TOKEN"] = apify_token
if openai_token:
    os.environ["OPENAI_API_KEY"] = openai_token

# Input for the search query
query = st.text_input("Enter your search query", value="latest developments in AI ethics")

# Run Analysis button
if st.button("Run Analysis"):
    # Check that API tokens have been provided
    if not (apify_token and openai_token):
        st.error("Please provide both API tokens to continue.")
    else:
        with st.spinner("Running analysis..."):
            try:
                analysis_result = research_topic(query)
                st.markdown("### Analysis Result")
                st.write(analysis_result)
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
