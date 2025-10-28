import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chisquare
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Charts will not be available. Please install with: pip install plotly")

def calculate_dice_theoretical(num_dice, dice_sides):
    if num_dice == 1:
        return {i: 1/dice_sides for i in range(1, dice_sides + 1)}

    from itertools import product
    min_sum = num_dice
    max_sum = num_dice * dice_sides

    sum_counts = {}
    total_outcomes = dice_sides ** num_dice

    for combination in product(range(1, dice_sides + 1), repeat=num_dice):
        total = sum(combination)
        sum_counts[total] = sum_counts.get(total, 0) + 1

    theoretical_probs = {}
    for sum_value in range(min_sum, max_sum + 1):
        count = sum_counts.get(sum_value, 0)
        theoretical_probs[sum_value] = count / total_outcomes

    return theoretical_probs

def calculate_coin_theoretical(bias=0.5):
    return {'Heads': bias, 'Tails': 1 - bias}

def perform_chi_square_test(observed_freq, theoretical_probs, num_trials):
    all_outcomes = sorted(set(observed_freq.keys()) | set(theoretical_probs.keys()))

    observed = []
    expected = []

    for outcome in all_outcomes:
        obs_count = observed_freq.get(outcome, 0)
        exp_prob = theoretical_probs.get(outcome, 0)
        exp_count = exp_prob * num_trials

        observed.append(obs_count)
        expected.append(exp_count)

    chi2_stat, p_value = chisquare(observed, expected)

    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': len(all_outcomes) - 1,
        'observed': observed,
        'expected': expected
    }

def create_basic_plot(observed_freq, theoretical_probs, num_trials, title):
    all_outcomes = sorted(set(observed_freq.keys()) | set(theoretical_probs.keys()))

    outcomes = [str(x) for x in all_outcomes]
    observed_counts = [observed_freq.get(outcome, 0) for outcome in all_outcomes]
    expected_counts = [theoretical_probs.get(outcome, 0) * num_trials for outcome in all_outcomes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=outcomes,
        y=observed_counts,
        name='Observed',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=outcomes,
        y=expected_counts,
        name='Expected',
        marker_color='red',
        opacity=0.7
    ))

    fig.update_layout(
        title=f'{title} - Observed vs Expected',
        xaxis_title='Outcome',
        yaxis_title='Count',
        barmode='group'
    )

    return fig

def plotly_fig_to_png(fig):
    fig_dict = fig.to_dict()

    plt.figure(figsize=(10, 6))

    outcomes = fig_dict['data'][0]['x']
    observed = fig_dict['data'][0]['y']
    expected = fig_dict['data'][1]['y']

    x_pos = np.arange(len(outcomes))
    width = 0.35

    plt.bar(x_pos - width/2, observed, width, label='Observed', color='blue')
    plt.bar(x_pos + width/2, expected, width, label='Expected', color='red', alpha=0.7)

    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.title(fig_dict['layout']['title']['text'])
    plt.xticks(x_pos, outcomes)
    plt.legend()
    plt.tight_layout()

    img_byte_arr = BytesIO()
    plt.savefig(img_byte_arr, format='png', dpi=150)
    img_byte_arr.seek(0)
    plt.close()

    return img_byte_arr.getvalue()

def create_dice_grid_image(display_values, cols, title="Dice Roll Results"):
    rows = int(np.ceil(len(display_values) / cols))

    cell_size = 50
    img_width = cols * cell_size + 20
    img_height = rows * cell_size + 50

    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    draw.text((10, 10), title, fill='black', font=title_font)

    value_index = 0
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_size + 10
            y1 = row * cell_size + 40
            x2 = x1 + cell_size - 5
            y2 = y1 + cell_size - 5

            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)

            if value_index < len(display_values):
                value = display_values[value_index]
                text_bbox = draw.textbbox((0, 0), str(value), font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = x1 + (cell_size - text_width) / 2 - 2
                text_y = y1 + (cell_size - text_height) / 2 - 2

                draw.text((text_x, text_y), str(value), fill='black', font=font)
                value_index += 1

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr.getvalue()

def create_dark_theme_table(df, title="Frequency Table"):
    rows, cols = df.shape
    cell_width, cell_height = 150, 40
    img_width = cell_width * (cols + 1)
    img_height = cell_height * (rows + 2)

    img = Image.new('RGB', (img_width, img_height), color='#2E3440')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        header_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    title_bbox = draw.textbbox((0, 0), title, font=header_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((img_width - title_width) / 2, 10), title, fill='#ECEFF4', font=header_font)

    for j, col in enumerate(df.columns):
        x1 = j * cell_width
        y1 = 40
        x2 = (j + 1) * cell_width
        y2 = 80

        draw.rectangle([x1, y1, x2, y2], outline='#4C566A', fill='#3B4252')

        text_bbox = draw.textbbox((0, 0), str(col), font=header_font)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((x1 + (cell_width - text_width) / 2, 45), str(col), fill='#ECEFF4', font=header_font)

    for i, (idx, row) in enumerate(df.iterrows()):
        for j, col in enumerate(df.columns):
            x1 = j * cell_width
            y1 = 80 + i * cell_height
            x2 = (j + 1) * cell_width
            y2 = y1 + cell_height

            if i % 2 == 0:
                cell_color = '#434C5E'
            else:
                cell_color = '#3B4252'

            draw.rectangle([x1, y1, x2, y2], outline='#4C566A', fill=cell_color)

            cell_value = str(row[col])

            text_bbox = draw.textbbox((0, 0), cell_value, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((x1 + (cell_width - text_width) / 2, y1 + 10), cell_value, fill='#ECEFF4', font=font)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr.getvalue()

def create_coin_grid_image(display_tosses, rows, cols, title="Coin Toss Results"):
    cell_size = 30
    img_width = cols * cell_size + 20
    img_height = rows * cell_size + 50

    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    draw.text((10, 10), title, fill='black', font=title_font)

    toss_index = 0
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_size + 10
            y1 = row * cell_size + 40
            x2 = x1 + cell_size - 5
            y2 = y1 + cell_size - 5

            if toss_index < len(display_tosses):
                result = display_tosses[toss_index]
                if result == 'Heads':
                    fill_color = '#4CAF50'
                    text_color = 'white'
                    text = 'H'
                else:
                    fill_color = '#FF5722'
                    text_color = 'white'
                    text = 'T'
            else:
                fill_color = '#f0f0f0'
                text_color = '#666'
                text = ''

            draw.rectangle([x1, y1, x2, y2], outline='black', fill=fill_color)

            if text:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = x1 + (cell_size - text_width) / 2 - 2
                text_y = y1 + (cell_size - text_height) / 2 - 2

                draw.text((text_x, text_y), text, fill=text_color, font=font)

            toss_index += 1

    draw.rectangle([10, img_height - 40, 30, img_height - 20], outline='black', fill='#4CAF50')
    draw.text((35, img_height - 35), "H = Heads", fill='black', font=font)

    draw.rectangle([120, img_height - 40, 140, img_height - 20], outline='black', fill='#FF5722')
    draw.text((145, img_height - 35), "T = Tails", fill='black', font=font)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr.getvalue()

st.set_page_config(page_title="Dice and Coin Simulation", page_icon="🎲", layout="wide")

st.title("  Probability Simulation  ")
st.title("[                   🎲      |      🪙                   ]")
st.write("Assignment : Probability Study of Dice and Coin Toss Simulations (Project 12) ")

if 'dice_results' not in st.session_state:
    st.session_state.dice_results = None
if 'coin_results' not in st.session_state:
    st.session_state.coin_results = None

tab1, tab2, tab3 = st.tabs(["🎲|Dice", "🪙|Coin", "📑|Theory"])

with tab1:
    st.header("Dice Roll Simulation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Settings")
        num_dice = st.selectbox("How many dice?", [1, 2], index=0)
        dice_sides = st.selectbox("Sides on each die", [4, 6, 8, 10, 12, 20], index=1)
        num_trials = st.slider("Number of rolls", min_value=10, max_value=100, step=10, value=10)

        if st.button(" Roll the Dice!", type="primary"):
            with st.spinner("Rolling dice..."):

                if num_dice == 1:
                    rolls = np.random.randint(1, dice_sides + 1, num_trials)
                    dice_values = [[roll] for roll in rolls]
                else:
                    dice1 = np.random.randint(1, dice_sides + 1, num_trials)
                    dice2 = np.random.randint(1, dice_sides + 1, num_trials)
                    rolls = dice1 + dice2
                    dice_values = [[d1, d2] for d1, d2 in zip(dice1, dice2)]

                unique_values, counts = np.unique(rolls, return_counts=True)
                observed_freq = dict(zip(unique_values, counts))

                theoretical_probs = calculate_dice_theoretical(num_dice, dice_sides)

                chi_result = perform_chi_square_test(observed_freq, theoretical_probs, num_trials)

                st.session_state.dice_results = {
                    'num_dice': num_dice,
                    'dice_sides': dice_sides,
                    'num_trials': num_trials,
                    'rolls': rolls,
                    'dice_values': dice_values,
                    'observed_freq': observed_freq,
                    'theoretical_probs': theoretical_probs,
                    'chi_square': chi_result
                }

        if st.session_state.dice_results is not None:


            results = st.session_state.dice_results
            dice_values = results['dice_values']
            num_trials = results['num_trials']

            if results['num_dice'] == 1:
                display_values = [dice_set[0] for dice_set in dice_values]
            else:
                display_values = []
                for dice_set in dice_values:
                    display_values.extend(dice_set)

            cols = 10
            rows = int(np.ceil(len(display_values) / cols))

            grid_html = f"""
            
            <div style='display: inline-block; border: 2px solid #333; padding: 15px; background-color: #f9f9f9;'>"""

            value_index = 0
            for row in range(rows):
                grid_html += "<div style='display: flex;'>"
                for col in range(cols):
                    if value_index < len(display_values):
                        face_value = display_values[value_index]
                        grid_html += f"<div style='width: 40px; height: 40px; border: 2px solid #333; background-color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px; color: #333; margin: 3px;'>{face_value}</div>"
                    else:
                        grid_html += f"<div style='width: 40px; height: 40px; border: 2px solid #ddd; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center; margin: 3px;'></div>"
                    value_index += 1
                grid_html += "</div>"

            grid_html += "</div>"

            st.markdown(grid_html, unsafe_allow_html=True)

            dice_grid_png = create_dice_grid_image(display_values, 10, "Dice Roll Results")
            st.download_button(
                label="Download Dice Grid as PNG",
                data=dice_grid_png,
                file_name="dice_grid.png",
                mime="image/png",
                help="Download the dice grid as a PNG image"
            )

            face_counts = {}
            for value in display_values:
                face_counts[value] = face_counts.get(value, 0) + 1




    with col2:
        if st.session_state.dice_results is not None:


            fig = create_basic_plot(results['observed_freq'], results['theoretical_probs'], 
                                  results['num_trials'], "Dice Results")
            st.plotly_chart(fig)

            png_data = plotly_fig_to_png(fig)
            st.download_button(
                label="Download Chart as PNG",
                data=png_data,
                file_name="dice_chart.png",
                mime="image/png",
                help="Download the chart as a PNG image"
            )

            st.subheader("Frequency Table")
            table_data = []
            for outcome in sorted(results['observed_freq'].keys()):
                observed = results['observed_freq'][outcome]
                expected = results['theoretical_probs'].get(outcome, 0) * results['num_trials']
                table_data.append({
                   'Outcome': outcome,
                   'Observed': observed,
                   'Expected': f"{expected:.1f}",
                   'Difference': f"{abs(observed - expected):.1f}"
           }) 

            df = pd.DataFrame(table_data)
            st.table(df)

            table_png = create_dark_theme_table(df, "Dice Frequency Table")
            st.download_button(
                label=" Download Table as PNG (Dark Theme)",
                data=table_png,
                file_name="frequency_table.png",
                mime="image/png",
                help="Download the frequency table as a dark-themed PNG image"
            )

with tab2:
    st.header("Coin Toss Simulation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Settings")
        coin_bias = st.slider("Probability of Heads", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        if coin_bias == 0.5:
            st.write("Fair coin")
        else:
            st.write(f"Biased coin (more likely to be {'heads' if coin_bias > 0.5 else 'tails'})")

        num_tosses = st.slider("Number of tosses", min_value=10, max_value=100, step=10, value=10)

        if st.button("Flip the Coin!", type="primary"):
            with st.spinner("Flipping coin..."):
                tosses = np.random.choice(['Heads', 'Tails'], size=num_tosses,
                                        p=[coin_bias, 1-coin_bias])

                unique_values, counts = np.unique(tosses, return_counts=True)
                observed_freq = dict(zip(unique_values, counts))

                if 'Heads' not in observed_freq:
                    observed_freq['Heads'] = 0
                if 'Tails' not in observed_freq:
                    observed_freq['Tails'] = 0

                theoretical_probs = calculate_coin_theoretical(coin_bias)

                chi_result = perform_chi_square_test(observed_freq, theoretical_probs, num_tosses)

                st.session_state.coin_results = {
                    'coin_bias': coin_bias,
                    'num_trials': num_tosses,
                    'observed_freq': observed_freq,
                    'theoretical_probs': theoretical_probs,
                    'chi_square': chi_result,
                    'tosses': tosses
                }


    with col2:
        if st.session_state.coin_results is not None:
            results = st.session_state.coin_results

            st.subheader("Results")

            heads_count = results['observed_freq'].get('Heads', 0)
            heads_percent = (heads_count / results['num_trials']) * 100

            chi_stat = results['chi_square']['statistic']
            p_value = results['chi_square']['p_value']

            col1_stat, col2_stat = st.columns(2)
            with col1_stat:
                st.metric("Heads Percentage", f"{heads_percent:.1f}%")
            with col2_stat:
                test_result = "PASS" if p_value > 0.05 else "FAIL"
                st.metric("Chi-Square Test", test_result)



            fig = create_basic_plot(results['observed_freq'], results['theoretical_probs'],
                                  results['num_trials'], "Coin Results")
            st.plotly_chart(fig)

            png_data = plotly_fig_to_png(fig)
            st.download_button(
                label="📊 Download Chart as PNG",
                data=png_data,
                file_name="coin_chart.png",
                mime="image/png",
                help="Download the chart as a PNG image"
            )

            st.subheader("Summary")
            table_data = []
            for outcome in ['Heads', 'Tails']:
                observed = results['observed_freq'][outcome]
                expected = results['theoretical_probs'][outcome] * results['num_trials']
                table_data.append({
                    'Outcome': outcome,
                    'Observed': observed,
                    'Expected': f"{expected:.1f}",
                    'Percentage': f"{(observed/results['num_trials']*100):.1f}%"
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df)

            table_png = create_dark_theme_table(df, "Coin Frequency Table")
            st.download_button(
                label="Download Table as PNG (Dark Theme)",
                data=table_png,
                file_name="coin_frequency_table.png",
                mime="image/png",
                help="Download the frequency table as a dark-themed PNG image"
            )

    if st.session_state.coin_results is not None:
        st.subheader(" Coin Grid Visualization")

        num_tosses = len(st.session_state.coin_results['tosses'])
        display_tosses = st.session_state.coin_results['tosses']

        if num_tosses <= 10:
            cols = min(5, num_tosses)
        elif num_tosses <= 20:
            cols = 5
        elif num_tosses <= 50:
            cols = 10
        else:
            cols = 10

        rows = int(np.ceil(num_tosses / cols))

        heads_count = sum(1 for toss in display_tosses if toss == 'Heads')
        tails_count = sum(1 for toss in display_tosses if toss == 'Tails')

        st.write(f"**Grid ({rows}×{cols}) - Total: {num_tosses} tosses**")
        st.write(f"🟢 **Heads:** {heads_count} ({heads_count/num_tosses*100:.1f}%) | 🔴 **Tails:** {tails_count} ({tails_count/num_tosses*100:.1f}%)")

        toss_index = 0

        for row in range(rows):
            grid_cols = st.columns(cols)

            for col_idx in range(cols):
                if toss_index < len(display_tosses):
                    result = display_tosses[toss_index]

                    with grid_cols[col_idx]:
                        if result == 'Heads':
                            st.success("H", icon="🟢")
                        else:
                            st.error("T", icon="🔴")

                    toss_index += 1
                else:
                    with grid_cols[col_idx]:
                        st.write("")

        st.write("")
        legend_col1, legend_col2 = st.columns(2)
        with legend_col1:
            st.write("🟢 **H = Heads**")
        with legend_col2:
            st.write("🔴 **T = Tails**")

        coin_grid_png = create_coin_grid_image(display_tosses, rows, cols, "Coin Toss Results")
        st.download_button(
            label=" Download Coin Grid as PNG",
            data=coin_grid_png,
            file_name="coin_grid.png",
            mime="image/png",
            help="Download the coin grid as a PNG image"
        )

with tab3: 
    st.title(" We will have to re-search theory on our own and add it up here.")        


st.markdown("---")