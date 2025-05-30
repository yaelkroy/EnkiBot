# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ==================================================================================================
# === EnkiBot Fact Extractor ===
# ==================================================================================================
# This module uses linguistic rules and morphological analysis to extract specific
# pieces of information from user text, such as a person's name mentioned in a query.
# This is faster and cheaper than using an LLM for simple, well-defined extraction tasks.
# ==================================================================================================

import logging
import re
import pymorphy3

logger = logging.getLogger(__name__)

# Initialize the morphological analyzer for Russian.
try:
    morph = pymorphy3.MorphAnalyzer()
except Exception as e:
    logger.error(f"Could not initialize pymorphy3 MorphAnalyzer: {e}. Fact extraction might fail.")
    morph = None

def find_user_search_query_in_text(text: str) -> str | None:
    """
    Analyzes text by lemmatizing words to their base form and looks for a combination
    of trigger words and prepositions to extract a user name.

    This is a classic Natural Language Processing (NLP) technique for entity extraction.

    Args:
        text: The user's message text.

    Returns:
        The extracted name as a string, or None if no name is found.
    """
    if not morph:
        logger.warning("Morphological analyzer not available. Skipping user search query extraction.")
        return None
        
    # Dictionaries of trigger word lemmas (base forms).
    # This makes the system robust to different word forms (e.g., 'tell', 'tells', 'told').
    TELL_LEMMAS = {'рассказать', 'поведать', 'сообщить', 'описать'}
    INFO_LEMMAS = {'информация', 'инфо', 'справка', 'досье', 'данные'}
    WHO_LEMMAS = {'кто', 'что'}
    EXPLAIN_LEMMAS = {'пояснить', 'объяснить'}
    REMEMBER_LEMMAS = {'помнить', 'напомнить'}

    # Prepositions that typically follow trigger words before a name.
    PREPOSITIONS = {'о', 'про', 'за', 'на', 'по'}

    # Split the text into words
    words = re.findall(r"[\w'-]+", text.lower())
    
    for i, word in enumerate(words):
        try:
            # Get the lemma (normal form) of the word
            lemma = morph.parse(word)[0].normal_form
            
            # Check if the lemma is one of our triggers
            is_trigger = (lemma in TELL_LEMMAS or
                          lemma in INFO_LEMMAS or
                          lemma in WHO_LEMMAS or
                          lemma in EXPLAIN_LEMMAS or
                          lemma in REMEMBER_LEMMAS)

            if is_trigger:
                # We found a trigger word. The name should follow it.
                start_index = i + 1
                
                # If the next word is a preposition, skip it.
                if start_index < len(words) and words[start_index] in PREPOSITIONS:
                    start_index += 1
                
                # Everything that follows (up to 3 words) is considered the name.
                if start_index < len(words):
                    # Capture 1 to 3 words after the trigger/preposition.
                    name_parts = words[start_index : start_index + 3]
                    extracted_name = " ".join(name_parts)
                    logger.info(f"Extracted potential user search query: '{extracted_name}'")
                    return extracted_name

        except Exception as e:
            logger.error(f"Error during lemmatization of word '{word}': {e}")
            continue
            
    return None