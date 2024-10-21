import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from os import path
import requests
import google.generativeai as genai
import os
import numpy as np 
import folium
from streamlit_folium import st_folium

# Load your DataFrame
df = pd.read_parquet(path.join("data", "processed_feedback_withwt.parquet"))

# Fill missing consolidated themes with the existing theme if only one is present
df['consolidated_theme_2_from_pivot'] = df['consolidated_theme_2_from_pivot'].fillna(df['consolidated_theme_1_from_pivot'])
df['consolidated_theme_3_from_pivot'] = df['consolidated_theme_3_from_pivot'].fillna(df['consolidated_theme_1_from_pivot'])
df['consolidated_theme_3_from_pivot'] = df['consolidated_theme_3_from_pivot'].fillna(df['consolidated_theme_2_from_pivot'])

# Repeat the same for raw themes if needed
df['raw_theme_2'] = df['raw_theme_2'].fillna(df['raw_theme_1'])
df['raw_theme_3'] = df['raw_theme_3'].fillna(df['raw_theme_1'])
df['raw_theme_3'] = df['raw_theme_3'].fillna(df['raw_theme_2'])

# Define the sentiment determination functions
def determine_sentiment(score):
    if score >= 8:
        return 'Positive'
    elif score >= 5:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment logic
df['sentiment_johndeere'] = df['recommend_johndeere_score'].apply(determine_sentiment)
df['sentiment_opcenter'] = df['recommend_opcenter_score'].apply(determine_sentiment)

df['product'] = df['product'].fillna('Others')

# Combine sentiments into a single column
def combine_sentiments(row):
    sentiments = [row['sentiment_johndeere'], row['sentiment_opcenter']]
    if 'Negative' in sentiments:
        return 'Negative'
    elif 'Neutral' in sentiments:
        return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df.apply(combine_sentiments, axis=1)

# Clean up consolidated theme columns by removing leading and trailing '**'
consolidated_theme_columns = [
    'consolidated_theme_1_from_pivot',
    'consolidated_theme_2_from_pivot',
    'consolidated_theme_3_from_pivot'
]

for col in consolidated_theme_columns:
    df[col] = df[col].astype(str).str.strip('*').replace('nan', pd.NA)

# Combine the importance factors by taking the maximum, ignoring NaNs
df['importance_factor'] = df[[
    'importance_factor_1_from_pivot',
    'importance_factor_2_from_pivot',
    'importance_factor_3_from_pivot'
]].max(axis=1)

# Fill NaN values in the new 'importance_factor' column with 0
df['importance_factor'] = df['importance_factor'].fillna(0)

# Replace NaN values with 'Other' in 'classification_category' and 'classification_focus'
df['classification_category'] = df['classification_category'].fillna('Other')
df['classification_focus'] = df['classification_focus'].fillna('Other')

# Convert 'last_submitted_timestamp' to datetime if necessary
df['last_submitted_timestamp'] = pd.to_datetime(df['last_submitted_timestamp'])

# Extract Year and Month for filtering
df['Year'] = df['last_submitted_timestamp'].dt.year
df['Month'] = df['last_submitted_timestamp'].dt.month

# Inject custom CSS for Trade Gothic Bold Extended
st.markdown(
    """
    <style>
    @font-face {
        font-family: 'Trade Gothic Bold Extended';
        src: url('fonts/trade-gothic-bold-extended.ttf') format('truetype');
    }
    .custom-title {
        font-family: 'Trade Gothic Bold Extended', sans-serif;
        color: #367C2B;
        font-size: 24px;
    }
    .custom-separator {
        font-size: 24px;
        padding-left: 175px;
        padding-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a header with the logo and the title
logo_url = "https://cdn.ux.deere.com/brand-foundations/1.4.0/logos/jd-logo.svg#green"

# Create columns for layout
col1, col2 = st.columns([1, 8])

with col1:
    st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center;">
        <span style="font-size: 32px; padding-left: 1px; padding-right: 0px;">|</span>
        <span style="color: #367C2B; font-size: 14px; font-weight: bold;">OpsInsight Ai</span>
    </div>
    """,
    unsafe_allow_html=True
)
with col2:
    st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn.ux.deere.com/brand-foundations/1.4.0/logos/jd-logo.svg#green" alt="Logo" width="550" user-select: none;">
    </div>
    """,
    unsafe_allow_html=True
)


# Display the "Not in Production" message at the top of the screen
st.markdown(
    """
    <div style="text-align: center; color: red; font-weight: bold; margin-top: 0; font-size: 20px;">
        ***Not in Production // ISG Hackathon 2024***
    </div>
    """,
    unsafe_allow_html=True
)

# Define a mapping of user_region to geographic coordinates (latitude and longitude)
region_coordinates = {
    '2': {'lat': 54.5260, 'lon': 15.2551},  # Europe
    '3': {'lat': -14.2350, 'lon': -51.9253},  # South America
    '4': {'lat': 37.0902, 'lon': -95.7129}  # North America
}

# Initialize session state for map visibility
if 'show_map' not in st.session_state:
    st.session_state['show_map'] = False

# Handle button clicks and update session state
with st.sidebar:
    st.markdown("<hr>", unsafe_allow_html=True)
    generate_summary = st.button("Generate Summary")
    analyze_relationships = st.button("Analyze Relationships")

# Add a "Reset" button at the top of the sidebar that refreshes the page
# Create a text link that acts as a reset button
st.sidebar.markdown(
    """
    <a href="javascript:window.location.reload(true);" style="color: #FF6B6B; font-weight: bold; text-decoration: none; font-size: 16px;">
        Reset
    </a>
    """,
    unsafe_allow_html=True
)
st.sidebar.header('Filter by Importance Factor')
min_importance = float(df['importance_factor'].min())
max_importance = float(df['importance_factor'].max())
importance_range = st.sidebar.slider(
    'Select Importance Factor Range',
    min_value=min_importance,
    max_value=max_importance,
    value=(min_importance, max_importance),
    step=0.1
)

# Apply filters to create df_filtered
df_filtered = df[
    (df['importance_factor'] >= importance_range[0]) &
    (df['importance_factor'] <= importance_range[1])
]

# Filter by Consolidated Theme
st.sidebar.header('Filter by Consolidated Theme')
consolidated_themes = pd.concat([
    df['consolidated_theme_1_from_pivot'],
    df['consolidated_theme_2_from_pivot'],
    df['consolidated_theme_3_from_pivot']
]).dropna().unique()
consolidated_themes = sorted(consolidated_themes)
consolidated_themes = ['All'] + consolidated_themes
selected_theme = st.sidebar.selectbox('Select Consolidated Theme', options=consolidated_themes)

if selected_theme != 'All':
    df_filtered = df_filtered[
        (df_filtered['consolidated_theme_1_from_pivot'] == selected_theme) |
        (df_filtered['consolidated_theme_2_from_pivot'] == selected_theme) |
        (df_filtered['consolidated_theme_3_from_pivot'] == selected_theme)
    ]


# Sentiment Filter
st.sidebar.header('Filter by Sentiment')
sentiment_options = df['sentiment'].unique()
selected_sentiments = st.sidebar.multiselect(
    'Select Sentiment(s)',
    options=sentiment_options,
    default=list(sentiment_options)
)
df_filtered = df_filtered[df_filtered['sentiment'].isin(selected_sentiments)]

# Add a dropdown filter for the 'product' column in the sidebar with a default value of "All"
st.sidebar.header('Filter by Product')
products = sorted(df['product'].unique())
products = ['All'] + products  # Add 'All' option to the beginning of the list

selected_product = st.sidebar.selectbox(
    'Select Product',
    options=products,
    index=0  # This sets "All" as the default selected option
)

# Apply the filter to create df_filtered
if selected_product != 'All':
    df_filtered = df_filtered[df_filtered['product'] == selected_product]

# Region Filter
st.sidebar.header('Filter by Region')
regions = df['user_region_fake'].sort_values().unique()
selected_regions = st.sidebar.multiselect(
    'Select Region(s)',
    options=regions,
    default=list(regions)
)
df_filtered = df_filtered[df_filtered['user_region_fake'].isin(selected_regions)]

# Classification Filters
st.sidebar.header('Filter by Classification Category')
classification_categories = df['classification_category'].sort_values().unique()
selected_categories = st.sidebar.multiselect(
    'Select Classification Category',
    options=classification_categories,
    default=list(classification_categories)
)
df_filtered = df_filtered[df_filtered['classification_category'].isin(selected_categories)]

# Year Filter
st.sidebar.header('Filter by Year')
years = df['Year'].sort_values().unique()
selected_years = st.sidebar.multiselect(
    'Select Year(s)',
    options=years,
    default=list(years)
)
if selected_years:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]

# Month Filter
st.sidebar.header('Filter by Month')
months = df['Month'].sort_values().unique()
selected_months = st.sidebar.multiselect(
    'Select Month(s)',
    options=months,
    default=list(months)
)
if selected_months:
    df_filtered = df_filtered[df_filtered['Month'].isin(selected_months)]


# Define the function to call the Gemini API
def gemini_generate(base_prompt, input_text):
    prompt = base_prompt + input_text
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return(response.text)
    
# Define the base prompt
base_prompt = (
    "You are an experienced AI bot, tasked with using the provided feedback to create an actionable, pointwise summary that highlights key pain points, suggestions, and areas of improvement for John Deere's Operations Center.\n"
    "Each feedback item is accompanied by a date and content. Use this information to identify trends or shifts in user experience over time, offering time-sensitive insights where applicable.\n"
    "Focus each point on insights that will help product managers discover opportunities for new features and enhancements.\n"
    "Present the summary as clear, concise bullet points without including any customer-specific information or identifiable details. Your goal is to distill actionable insights that will inform product development and feature updates.\n"
    "Your output should include points related to:\n"
    "- Pain points\n"
    "- Opportunities\n"
    "- Customer behavior insights\n"
    "- Requested features\n"
    "- Areas needing improvement\n"
)

# Analyze Relationships button functionality
if analyze_relationships:
    st.markdown("<h2 style='text-align: center;'>Analyzing Relationships</h2>", unsafe_allow_html=True)
    
    # Prepare data for Sankey diagram - Consolidated Themes
    theme_pairs = pd.DataFrame()
    for index, row in df_filtered.iterrows():
        consolidated_themes = list(set([
            row['consolidated_theme_1_from_pivot'],
            row['consolidated_theme_2_from_pivot'],
            row['consolidated_theme_3_from_pivot']
        ]))
        for i, source in enumerate(consolidated_themes):
            for target in consolidated_themes[i+1:]:
                theme_pairs = pd.concat([theme_pairs, pd.DataFrame({'source': [source], 'target': [target]})], ignore_index=True)

    # Aggregate the theme pairs to get counts for each flow
    theme_flows = theme_pairs.groupby(['source', 'target']).size().reset_index(name='value')

    # Sort by the most frequent connections and keep only the top 10
    top_theme_flows = theme_flows.sort_values(by='value', ascending=False).head(10)

    # Create unique labels from sources and targets for the Sankey diagram
    all_labels = list(pd.concat([top_theme_flows['source'], top_theme_flows['target']]).unique())

    # Map labels to indices for the Sankey diagram
    label_to_index = {label: i for i, label in enumerate(all_labels)}

    # Map sources and targets to their indices
    top_theme_flows['source_idx'] = top_theme_flows['source'].map(label_to_index)
    top_theme_flows['target_idx'] = top_theme_flows['target'].map(label_to_index)

    # Create the Sankey diagram with the top 10 connections - Consolidated Themes
    sankey_figure_consolidated = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,  
            thickness=20,  
            line=dict(color="black", width=0.5),
            label=all_labels,
            color="blue"
        ),
        link=dict(
            source=top_theme_flows['source_idx'],
            target=top_theme_flows['target_idx'],
            value=top_theme_flows['value'],
            color="rgba(31, 119, 180, 0.5)"
        )
    )])

    # Update layout for better presentation
    sankey_figure_consolidated.update_layout(
        title_text="Top 10 Relationships Between Consolidated Themes",
        font_size=14,
        width=900,
        height=600,
    )

    # Display the Sankey diagram in Streamlit
    st.plotly_chart(sankey_figure_consolidated, use_container_width=True)
    
    # Prepare data for Sankey diagram - Raw Themes
    raw_theme_pairs = pd.DataFrame()
    for index, row in df_filtered.iterrows():
        raw_themes = list(set([
            row['raw_theme_1'],
            row['raw_theme_2'],
            row['raw_theme_3']
        ]))
        for i, source in enumerate(raw_themes):
            for target in raw_themes[i+1:]:
                raw_theme_pairs = pd.concat([raw_theme_pairs, pd.DataFrame({'source': [source], 'target': [target]})], ignore_index=True)

    # Aggregate the raw theme pairs to get counts for each flow
    raw_theme_flows = raw_theme_pairs.groupby(['source', 'target']).size().reset_index(name='value')

    # Sort by the most frequent connections and keep only the top 10
    top_raw_theme_flows = raw_theme_flows.sort_values(by='value', ascending=False).head(10)

    # Create unique labels from sources and targets for the Sankey diagram - Raw Themes
    all_raw_labels = list(pd.concat([top_raw_theme_flows['source'], top_raw_theme_flows['target']]).unique())

    # Map labels to indices for the Sankey diagram - Raw Themes
    label_to_raw_index = {label: i for i, label in enumerate(all_raw_labels)}

    # Map sources and targets to their indices - Raw Themes
    top_raw_theme_flows['source_idx'] = top_raw_theme_flows['source'].map(label_to_raw_index)
    top_raw_theme_flows['target_idx'] = top_raw_theme_flows['target'].map(label_to_raw_index)

    # Create the Sankey diagram with the top 10 connections - Raw Themes
    sankey_figure_raw = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,  
            thickness=20,  
            line=dict(color="black", width=0.5),
            label=all_raw_labels,
            color="green"
        ),
        link=dict(
            source=top_raw_theme_flows['source_idx'],
            target=top_raw_theme_flows['target_idx'],
            value=top_raw_theme_flows['value'],
            color="rgba(0, 128, 0, 0.5)"
        )
    )])

    # Update layout for better presentation
    sankey_figure_raw.update_layout(
        title_text="Top 10 Relationships Between Raw Themes",
        font_size=14,
        width=900,
        height=600,
    )

    # Display the Sankey diagram for raw themes in Streamlit
    st.plotly_chart(sankey_figure_raw, use_container_width=True)

# Generate Summary button functionality
elif generate_summary:
    st.markdown("<h2 style='text-align: center;'>AI Summary</h2>", unsafe_allow_html=True)
    
    # Extract the list of translated comments and their associated timestamps from the filtered DataFrame
    feedback_data = df_filtered[['translated_comment', 'last_submitted_timestamp']].dropna(subset=['translated_comment'])
    
    # Display a message if no comments are available
    if feedback_data.empty:
        st.write("No feedback available for generating the summary.")
    else:
        # Prepare the input text with timestamps
        input_text = "\n".join(
            f"Date: {row['last_submitted_timestamp'].strftime('%Y-%m-%d')}, Feedback: {row['translated_comment']}"
            for _, row in feedback_data.iterrows()
        )
        
        # Call the Gemini API with the prompt and text
        st.write("Generating summary, please wait...")
        summary = gemini_generate(base_prompt, input_text)
        
        # Display the generated summary
        st.write(summary)

else:
    # Show bar chart for top themes and raw themes (if a specific theme is selected)
    st.header('Categories of Feedback')
    
    # Prepare the data for the stacked bar chart for consolidated themes
    theme_trends = df_filtered.melt(
        id_vars=['sentiment'],
        value_vars=[
            'consolidated_theme_1_from_pivot',
            'consolidated_theme_2_from_pivot',
            'consolidated_theme_3_from_pivot'
        ],
        var_name='theme_type',
        value_name='theme'
    ).dropna()

    # Exclude rows where 'theme' is None or empty
    theme_trends = theme_trends[theme_trends['theme'].notna() & (theme_trends['theme'] != 'None')]

    # Get the top 10 themes by occurrence
    top_themes = theme_trends['theme'].value_counts().nlargest(10).index
    filtered_themes = theme_trends[theme_trends['theme'].isin(top_themes)]

    # Aggregate counts of sentiments for each theme
    theme_sentiment_counts = filtered_themes.groupby(['theme', 'sentiment']).size().reset_index(name='count')

    # Create a stacked bar chart using Altair for consolidated themes
    consolidated_chart = alt.Chart(theme_sentiment_counts).mark_bar().encode(
        x=alt.X('theme:N', sort='-y', title=None, axis=alt.Axis(labels=False)),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color(
            'sentiment:N',
            scale=alt.Scale(
                domain=['Positive', 'Neutral', 'Negative'],
                range=['#4CAF50', '#FFEB3B', '#F44336']
            )
        ),
        tooltip=['theme', 'sentiment', 'count']
    ).properties(
        title='Top 10 Categories of Feedback',
        width=800,
        height=400
    )

    st.altair_chart(consolidated_chart)

    # Display additional chart if a specific theme is selected
    if selected_theme != 'All':
        raw_theme_trends = df_filtered.melt(
            id_vars=['sentiment'],
            value_vars=['raw_theme_1', 'raw_theme_2', 'raw_theme_3'],
            var_name='raw_theme_type',
            value_name='raw_theme'
        ).dropna()

        # Exclude rows where 'raw_theme' is None or empty
        raw_theme_trends = raw_theme_trends[
            raw_theme_trends['raw_theme'].notna() & (raw_theme_trends['raw_theme'] != 'None')
        ]

        # Get the top 10 raw themes by occurrence
        top_raw_themes = raw_theme_trends['raw_theme'].value_counts().nlargest(10).index
        filtered_raw_themes = raw_theme_trends[raw_theme_trends['raw_theme'].isin(top_raw_themes)]

        # Aggregate counts of sentiments for each raw theme
        raw_theme_sentiment_counts = filtered_raw_themes.groupby(['raw_theme', 'sentiment']).size().reset_index(name='count')

        # Create a stacked bar chart using Altair for raw themes
        raw_chart = alt.Chart(raw_theme_sentiment_counts).mark_bar().encode(
            x=alt.X('raw_theme:N', sort='-y', title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color(
                'sentiment:N',
                scale=alt.Scale(
                    domain=['Positive', 'Neutral', 'Negative'],
                    range=['#4CAF50', '#FFEB3B', '#F44336']
                )
            ),
            tooltip=['raw_theme', 'sentiment', 'count']
        ).properties(
            title=f'Top 10 Feedback Themes for Category {selected_theme}',
            width=800,
            height=400
        )

        # Display the raw themes plot above the consolidated theme plot
        st.altair_chart(raw_chart)

    # Display the filtered feedback data table at the bottom
    st.header('Filtered Feedback Data')
    if df_filtered.empty:
        st.write('No feedback data available for the selected filters.')
    else:
        st.dataframe(df_filtered[[
            'last_submitted_timestamp', 'translated_comment', 'importance_factor',
            'classification_category', 'classification_focus', 'user_region',
            'raw_theme_1', 'raw_theme_2', 'raw_theme_3',
            'consolidated_theme_1_from_pivot', 'consolidated_theme_2_from_pivot',
            'consolidated_theme_3_from_pivot', 'recommend_johndeere_score',
       'recommend_opcenter_score', 'product', 'browser', 'device',
       'operating_system'
        ]].reset_index(drop=True))


 # Line plot for feedback counts over time, segmented by importance factors
line_data = df_filtered.copy()
line_data['importance_segment'] = pd.cut(
    line_data['importance_factor'],
    bins=[0, 4, 7, 11],
    labels=['0-4', '4-7', '7-11']
)

# Create a 'YearMonth' column for better date representation
line_data['YearMonth'] = pd.to_datetime(line_data['Year'].astype(str) + '-' + line_data['Month'].astype(str))
st.write("Number of rows in data:", len(line_data))
# Aggregate data for the line plot
line_data_agg = line_data.groupby(['YearMonth', 'importance_segment']).size().reset_index(name='count')

# Sort the data by 'YearMonth' to ensure correct ordering
line_data_agg = line_data_agg.sort_values('YearMonth')

# Create a line chart using Altair
if not line_data_agg.empty:
    line_chart = alt.Chart(line_data_agg).mark_line().encode(
        x=alt.X('YearMonth:T', title='Year-Month', axis=alt.Axis(format='%b %Y', labelAngle=-45)),  # Display month and year
        y=alt.Y('count:Q', title='Feedback Count'),
        color=alt.Color('importance_segment:N', title='Importance Factor Segment'),
        tooltip=[alt.Tooltip('YearMonth:T', title='Year-Month', format='%b %Y'), 'importance_segment', 'count']
    ).properties(
        title='Feedback Count Over Time by Importance Factor',
        width=800,
        height=400
    )

    # Display the line chart
    st.altair_chart(line_chart)

   # Add map
    st.markdown("<h4 style='text-align: center;'>User Region Feedback Map</h4>", unsafe_allow_html=True)
    st.markdown("<h7 style='text-align: center;'>Placeholder values</h7>", unsafe_allow_html=True)

    # Calculate the counts of feedback per user_region based on filtered data
    feedback_counts = df_filtered['user_region_fake'].value_counts().reset_index()
    feedback_counts.columns = ['user_region_fake', 'count']

    # Map regions to their respective latitude and longitude
    feedback_counts['lat'] = feedback_counts['user_region_fake'].map(lambda x: region_coordinates.get(x, {'lat': 0, 'lon': 0})['lat'])
    feedback_counts['lon'] = feedback_counts['user_region_fake'].map(lambda x: region_coordinates.get(x, {'lat': 0, 'lon': 0})['lon'])

    # Define min and max circle radius for scaling
    min_radius = 5
    max_radius = 30

    # Normalize the counts to scale the circle size between min_radius and max_radius
    if len(feedback_counts) > 0 and feedback_counts['count'].max() > 0:
        max_count = feedback_counts['count'].max()
        feedback_counts['scaled_radius'] = feedback_counts['count'].apply(
            lambda x: min_radius + (x / max_count) * (max_radius - min_radius)
        )
    else:
        feedback_counts['scaled_radius'] = min_radius

    # Create a base Folium map with a dark theme
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB dark_matter', control_scale=True)

    # Add circles to the map for each region
    for _, row in feedback_counts.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=row['scaled_radius'],  # Adjust the circle size based on scaled radius
            color='#367C2B',  # John Deere green for the circle
            fill=True,
            fill_color='#367C2B',
            fill_opacity=0.5,
            tooltip=f"Region: {row['user_region_fake']}, Feedback Count: {row['count']}"
        ).add_to(m)

    # Display the map using Streamlit-Folium
    st_folium(m, width=800, height=500)

else:
    st.write("No data available for the selected date range and importance segments.")



# Add a footer at the very bottom of the page with custom styling
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        background-color: transparent;
        padding: 5px;
        font-size: 12px;
        color: #367C2B; /* John Deere green color */
    }
    </style>
    <div class="footer">
        ISG Hackathon 2024 Entry by, <a href="mailto:hakimmoazamiqbal@johndeere.org" style="color: #367C2B; text-decoration: none;">Moazam Hakim</a>
    </div>
    """,
    unsafe_allow_html=True
)


