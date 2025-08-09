{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM2n/vaAjw9WzqTrpaiqGv+"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HcY_0cSUGRl-"
      },
      "outputs": [],
      "source": [
        "\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "st.set_page_config(page_title=\"Imputation Method Recommender\", layout=\"centered\")\n",
        "\n",
        "st.title(\"Imputation Method Recommender\")\n",
        "st.markdown(\"Upload a CSV dataset and receive a recommendation for the best imputation technique based on its features.\")\n",
        "\n",
        "uploaded_file = st.file_uploader(\"Upload your CSV dataset\", type=[\"csv\"])\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    df = pd.read_csv(uploaded_file)\n",
        "\n",
        "    # To analyze dataset characteristics\n",
        "    num_features = df.select_dtypes(include=[np.number]).shape[1]\n",
        "    cat_features = df.select_dtypes(exclude=[np.number]).shape[1]\n",
        "    missing_rate = df.isnull().mean().mean()\n",
        "\n",
        "    st.subheader(\"📊 Dataset Summary\")\n",
        "    st.write({\n",
        "        \"Samples\": df.shape[0],\n",
        "        \"Total Features\": df.shape[1],\n",
        "        \"Numerical Features\": num_features,\n",
        "        \"Categorical Features\": cat_features,\n",
        "        \"Missing Rate (%)\": round(missing_rate * 100, 2)\n",
        "    })\n",
        "\n",
        "    # Recommendation engine\n",
        "    def recommend_method(missing, num_feat, cat_feat):\n",
        "        if missing < 0.05:\n",
        "            return \"KNN or Mean — work well for low missingness.\"\n",
        "        elif cat_feat > num_feat:\n",
        "            return \"MICE — good for categorical-rich datasets.\"\n",
        "        elif num_feat > 20:\n",
        "            return \"Autoencoder — good performance for complex, high-dimensional data.\"\n",
        "        else:\n",
        "            return \"MICE or KNN — maintains performance in most clinical datasets.\"\n",
        "\n",
        "    recommendation = recommend_method(missing_rate, num_features, cat_features)\n",
        "\n",
        "    st.success(\"Recommended Imputation Method:\")\n",
        "    st.write(recommendation)\n"
      ]
    }
  ]
}