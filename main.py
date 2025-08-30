import gradio as gr
import Bayes
import NLP
from BayesClass import CustomNaiveBayesClassifier


def prediction(text):
    finalprobabilities = {}
    bayesresult = Bayes.Bayes(text)
    nlpresult = NLP.nlp(text)
    finalprobabilities['valid'] = bayesresult['custom_model']['probabilities']['valid'] * 0.20 + bayesresult['sklearn_model']['probabilities']['valid'] * 0.20 + nlpresult[0]['all_probabilities']['valid'] * 0.60
    finalprobabilities['irrelevant'] = bayesresult['custom_model']['probabilities']['irrelevant'] * 0.20 + bayesresult['sklearn_model']['probabilities']['irrelevant'] * 0.20 + nlpresult[0]['all_probabilities']['irrelevant'] * 0.60
    finalprobabilities['advertisement'] = bayesresult['custom_model']['probabilities']['advertisement'] * 0.20 + bayesresult['sklearn_model']['probabilities']['advertisement'] * 0.20 + nlpresult[0]['all_probabilities']['advertisement'] * 0.60
    finalprobabilities['rant_without_visit'] = bayesresult['custom_model']['probabilities']['rant_without_visit'] * 0.20 + bayesresult['sklearn_model']['probabilities']['rant_without_visit'] * 0.20 + nlpresult[0]['all_probabilities']['rant_without_visit'] * 0.60
    return finalprobabilities

def classify_single_review(text):
    """Gradio interface function for single review classification."""
    if not text or not text.strip():
        return "Please enter a review to classify", "No text provided"
    
    result = prediction(text)
    
    # Format the output
    if result:
        confidence_text = "\n".join([
            f"{label}: {score:.3f}" 
            for label, score in sorted(result.items(), 
                                     key=lambda x: x[1], reverse=True)
        ])
        return ' '.join(str(max(result, key=result.get)).split("_")).capitalize(), confidence_text
    else:
        return result, "No confidence scores available"

def classify_batch_reviews(texts):
    """Gradio interface function for batch review classification."""
    if not texts or not texts.strip():
        return "Please enter reviews to classify (one per line)"
    
    results = []

    for i in texts.split("\n"):
        result = prediction(i)
        results.append(' '.join(str(max(result, key=result.get)).split("_")).capitalize())
    return '\n'.join(results)

# Create Gradio interface
with gr.Blocks(title="Restaurant Review Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Restaurant Review Classifier
        
        This tool classifies restaurant reviews into four categories:
        - **Valid**: Genuine reviews with useful information
        - **Advertisement**: Promotional content
        - **Rant without visit**: Complaints without actual restaurant visit
        - **Irrelevant**: Off-topic or unrelated content
        
        Choose between single review analysis or batch processing below.
        """
    )
    
    with gr.Tabs():
        # Single Review Tab
        with gr.TabItem("Single Review Analysis"):
            with gr.Row():
                with gr.Column():
                    single_input = gr.Textbox(
                        label="Restaurant Review Text",
                        placeholder="Enter a restaurant review here...",
                        lines=4,
                        max_lines=10
                    )
                    single_button = gr.Button("Classify Review", variant="primary")
                
                with gr.Column():
                    single_prediction = gr.Textbox(
                        label="Predicted Category",
                        interactive=False
                    )
                    single_confidence = gr.Textbox(
                        label="Confidence Scores",
                        lines=4,
                        interactive=False
                    )
            
            # Example reviews
            gr.Examples(
                examples=[
                    ["The food was absolutely amazing! The pasta was perfectly cooked and the service was excellent. Highly recommend this place for a romantic dinner."],
                    ["Check out our new restaurant! 50% off all meals this weekend! Visit www.example.com for more deals and promotions."],
                    ["This place is terrible! I've never been there but I heard from my friend's cousin that the food is bad."],
                    ["The weather is nice today. I like cats. My car needs new tires. Random thoughts about nothing related to restaurants."]
                ],
                inputs=[single_input]
            )
        
        # Batch Processing Tab
        with gr.TabItem("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        label="Multiple Reviews (one per line)",
                        placeholder="Enter multiple reviews, one per line...\n\nExample:\nGreat food and service!\nThis is just an ad for our restaurant.\nNever been there but heard it's bad.",
                        lines=8,
                        max_lines=20
                    )
                    batch_button = gr.Button("Classify All Reviews", variant="primary")
                
                with gr.Column():
                    batch_output = gr.Textbox(
                        label="Classification Results",
                        lines=15,
                        max_lines=25,
                        interactive=False
                    )
    
    # Model Information
    with gr.Accordion("Model Information", open=False):
        model_info = f"""
        ### Available Models
        - **Custom Naive Bayes:** ✅ Loaded
        - **Sklearn + TF-IDF:** ✅ Loaded
        - **NLP Analysis:** ✅ Loaded
        
        ### Model Details
        - **Ensemble Approach**: Combines multiple models for better accuracy
        - **Weights**: Custom Naive Bayes (20%), Sklearn Naive Bayes + TF-IDF (20%), NLP (60%)
        - **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Categories**: 4 classes (valid, advertisement, rant_without_visit, irrelevant)
        - **Language**: English text processing with stop word removal
        - **NLP Features**: Sentiment analysis, keyword matching, pattern recognition
        
        ### Usage Tips
        - The model works best with restaurant-related text
        - Longer, more detailed reviews generally produce more accurate classifications
        - The confidence scores show the probability for each category
        - Ensemble prediction combines all available models for optimal results
        """
        gr.Markdown(model_info)
    
    # Event handlers
    single_button.click(
        fn=classify_single_review,
        inputs=[single_input],
        outputs=[single_prediction, single_confidence]
    )
    
    batch_button.click(
        fn=classify_batch_reviews,
        inputs=[batch_input],
        outputs=[batch_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        share=True,  # Set to True if you want a public link
        show_error=True
    )