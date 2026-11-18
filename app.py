import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

st.set_page_config(page_title='Food Delivery – Customer Retention', layout='centered')
st.title('🍔 Food Delivery – Customer Inactivity Risk')

DATA_FILE = 'Fooddelivery_clean.csv'

SEGMENTATION_COLS = ['order_value', 'order_frequency', 'loyalty_points']

SEGMENT_STRATEGY = {
    'High Spenders': 'Protect high-value orders: priority delivery, free delivery on large orders, '
                     'premium bundles and proactive compensation for any delay.',
    'Loyal Regulars': 'Reward and retain: VIP tier, early access to new restaurants, '
                      'referral bonuses and birthday rewards.',
    'Unrewarded Regulars': 'They order often but hold few points: run double-points weeks, '
                           'show the points balance after each order and enrol them in the loyalty tier.',
    'Dormant Point Holders': 'They have many points but order rarely: send points-expiry reminders '
                             'and "redeem your points" offers to bring them back.',
    'Casual Low-Engagement': 'Build the ordering habit: discounts on the next 3 orders, '
                             'personalised recommendations and reminder notifications.'
}

DUMMY_COLS = ['gender', 'age', 'city', 'restaurant_name', 'dish_name',
              'payment_method', 'delivery_status', 'segment']


def name_segment(row):
    if row['order_value'] >= 1.5:
        return 'High Spenders'
    frequent = row['order_frequency'] >= 1
    loyal = row['loyalty_points'] >= 1
    if frequent and loyal:
        return 'Loyal Regulars'
    if frequent and not loyal:
        return 'Unrewarded Regulars'
    if not frequent and loyal:
        return 'Dormant Point Holders'
    return 'Casual Low-Engagement'


# ----------------------------------------------------------
# Train the models when the app starts (cached, runs once).
# Training here avoids loading .pkl files, so the app works
# with whatever library versions Streamlit Cloud installs.
# ----------------------------------------------------------
@st.cache_resource(show_spinner='Training models on the clean data...')
def train_models():

    df = pd.read_csv(DATA_FILE)
    df['rating'] = df['rating'].fillna(0)

    # ---------- K-Means segmentation (same as the notebook) ----------
    seg_scaler = StandardScaler()
    X_seg = seg_scaler.fit_transform(df[SEGMENTATION_COLS])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_seg)

    cluster_profile = df.groupby('cluster')[SEGMENTATION_COLS].mean()
    cluster_vs_overall = cluster_profile / df[SEGMENTATION_COLS].mean()

    segment_names = {}
    for cluster, row in cluster_vs_overall.iterrows():
        name = name_segment(row)
        if name in segment_names.values():
            name = f'{name} {cluster}'
        segment_names[int(cluster)] = name

    df['segment'] = df['cluster'].map(segment_names)

    # ---------- Constrained gradient boosting (same as the notebook) ----------
    df_model = df.drop(['customer_id', 'last_order_date', 'days_since_last_order',
                        'signup_date', 'order_date', 'rating_date', 'category', 'cluster'], axis=1)

    df_encoded = pd.get_dummies(df_model, columns=DUMMY_COLS, drop_first=True,
                                prefix=DUMMY_COLS, dtype=int)

    feature_cols = [c for c in df_encoded.columns if c != 'churned']

    x = df_encoded[feature_cols]
    y = df_encoded['churned']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    constraints = {
        'order_frequency': -1,
        'loyalty_points': -1,
        'rating': -1,
        'delivery_status_Delayed': -1,
        'delivery_status_Delivered': -1
    }
    monotonic = [constraints.get(c, 0) for c in feature_cols]

    model = HistGradientBoostingClassifier(
        monotonic_cst=monotonic,
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        min_samples_leaf=40,
        class_weight='balanced',
        random_state=42)

    model.fit(x_train, y_train)

    info = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
        'baseline_accuracy': accuracy_score(y_test, np.zeros(len(y_test))),
        'inactive_rate': df['churned'].mean()
    }

    return model, feature_cols, kmeans, seg_scaler, segment_names, info


if not os.path.exists(DATA_FILE):
    st.error(f'❌ {DATA_FILE} not found. Upload it to the same GitHub folder as app.py.')
    st.stop()

model, feature_cols, kmeans, seg_scaler, segment_names, info = train_models()

st.success('✅ Models trained successfully')
st.caption(f"Model: Constrained Gradient Boosting | ROC-AUC: {info['roc_auc']:.2f} | "
           f"Baseline accuracy: {info['baseline_accuracy']*100:.1f}% | "
           f"Overall inactive rate: {info['inactive_rate']*100:.1f}%")

restaurant_menu = {
    "McDonald's": ['Burger', 'Fries'],
    'Burger King': ['Burger', 'Fries'],
    'KFC': ['Burger', 'Fries'],
    'Subway': ['Sandwich'],
    'Pizza Hut': ['Pizza', 'Pasta']
}

# ----------------------------------------------------------
# Input section
# ----------------------------------------------------------
st.header('Enter Customer and Order Details')
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
    age = st.selectbox('Age Group', ['Teenager', 'Adult', 'Senior'])
    city = st.selectbox('City', ['Islamabad', 'Karachi', 'Lahore', 'Multan', 'Peshawar'])
    restaurant = st.selectbox('Restaurant', list(restaurant_menu.keys()))
    dish = st.selectbox('Dish', restaurant_menu[restaurant])
    payment = st.selectbox('Payment Method', ['Card', 'Cash', 'Wallet'])

with col2:
    delivery = st.selectbox('Delivery Status', ['Delivered', 'Delayed', 'Cancelled'])
    quantity = st.slider('Quantity', 1, 5, 2)
    price = st.number_input('Price per item', min_value=100.0, max_value=1500.0, value=800.0)
    order_frequency = st.slider('Order Frequency', 1, 50, 25)
    loyalty_points = st.slider('Loyalty Points', 0, 500, 250)
    if delivery == 'Cancelled':
        rating = 0
        st.info('Cancelled orders cannot be rated')
    else:
        rating = st.slider('Rating', 1, 5, 3)

order_value = quantity * price

# ----------------------------------------------------------
# Customer segment from K-Means
# ----------------------------------------------------------
seg_input = pd.DataFrame([[order_value, order_frequency, loyalty_points]], columns=SEGMENTATION_COLS)
cluster = int(kmeans.predict(seg_scaler.transform(seg_input))[0])
segment = segment_names[cluster]
segment_text = SEGMENT_STRATEGY.get(segment, SEGMENT_STRATEGY.get(segment.rsplit(' ', 1)[0], ''))

input_dict = {
    'quantity': quantity,
    'price': price,
    'order_frequency': order_frequency,
    'loyalty_points': loyalty_points,
    'rating': rating,
    'order_value': order_value,
    f'gender_{gender}': 1,
    f'age_{age}': 1,
    f'city_{city}': 1,
    f'restaurant_name_{restaurant}': 1,
    f'dish_name_{dish}': 1,
    f'payment_method_{payment}': 1,
    f'delivery_status_{delivery}': 1,
    f'segment_{segment}': 1
}

input_df = pd.DataFrame([input_dict])

st.subheader('Input Preview')
st.dataframe(pd.DataFrame([{
    'Gender': gender, 'Age': age, 'City': city, 'Restaurant': restaurant, 'Dish': dish,
    'Payment': payment, 'Delivery': delivery, 'Quantity': quantity, 'Price': price,
    'Order Value': order_value, 'Order Frequency': order_frequency,
    'Loyalty Points': loyalty_points, 'Rating': rating
}]))

st.subheader(f'Customer Segment: {segment}')
st.write(segment_text)

# ----------------------------------------------------------
# Business-rule score (expert rules, 0 = no risk, 1 = maximum risk)
# ----------------------------------------------------------
MODEL_WEIGHT = 0.5
RULE_WEIGHT = 0.5


def business_rule_score():
    frequency_risk = (50 - order_frequency) / 49
    loyalty_risk = (500 - loyalty_points) / 500
    rating_risk = 1.0 if rating == 0 else (5 - rating) / 4
    delivery_risk = {'Cancelled': 1.0, 'Delayed': 0.6, 'Delivered': 0.0}[delivery]

    parts = {
        'Order frequency (35%)': 0.35 * frequency_risk,
        'Delivery experience (25%)': 0.25 * delivery_risk,
        'Rating (20%)': 0.20 * rating_risk,
        'Loyalty points (20%)': 0.20 * loyalty_risk
    }
    return sum(parts.values()), parts


def retention_actions(prob):
    actions = []
    if delivery == 'Cancelled':
        actions.append('Send an apology and a full-value credit for the cancelled order.')
    elif delivery == 'Delayed':
        actions.append('Send an apology with a delivery-fee waiver on the next order.')
    if rating != 0 and rating <= 2:
        actions.append('Follow up on the low rating and share it with the restaurant partner.')
    if loyalty_points < 100:
        actions.append('Offer bonus loyalty points on the next order to build engagement.')
    elif loyalty_points >= 300:
        actions.append('Remind the customer of the loyalty points they can redeem.')
    if order_frequency < 10:
        actions.append('Send a first-orders bundle discount to build an ordering habit.')
    if prob >= 0.5:
        actions.append('Add to the win-back campaign: reminder at 30 days, discount at 60 days.')
    if not actions:
        actions.append('Keep in the regular engagement program (personalised recommendations).')
    return actions


if st.button('🔍 Predict Inactivity Risk'):

    X_pred = input_df.reindex(columns=feature_cols, fill_value=0)

    model_prob = model.predict_proba(X_pred)[0][1]
    rule_score, parts = business_rule_score()

    prob = MODEL_WEIGHT * model_prob + RULE_WEIGHT * rule_score

    if prob >= 0.6:
        st.error(f'⚠️ High risk of becoming inactive (Prob: {prob:.2f})')
    elif prob >= 0.4:
        st.warning(f'🟠 Medium risk of becoming inactive (Prob: {prob:.2f})')
    else:
        st.success(f'✅ Low risk of becoming inactive (Prob: {prob:.2f})')

    c1, c2, c3 = st.columns(3)
    c1.metric('Model score', f'{model_prob:.2f}')
    c2.metric('Business-rule score', f'{rule_score:.2f}')
    c3.metric('Final risk', f'{prob:.2f}')

    st.subheader('Business-rule breakdown')
    st.dataframe(pd.DataFrame({'Contribution to rule score': parts}).round(3))

    st.subheader('Suggested Retention Actions')
    st.write(f'• Segment offer ({segment}): ' + segment_text)
    for action in retention_actions(prob):
        st.write('• ' + action)

    st.caption('Final risk = 50% model score + 50% business-rule score. The model is a '
               'monotonic-constrained gradient boosting model trained on synthetic data with '
               'limited predictive power; the business rules are expert assumptions.')
