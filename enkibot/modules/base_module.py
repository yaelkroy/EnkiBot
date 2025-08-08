# enkibot/modules/base_module.py
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
# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# === EnkiBot Base Module ===
# ==================================================================================================
# This file is intended to hold an Abstract Base Class (ABC) for all functional modules.
# In a future evolution, all modules (e.g., IntentRecognizer, FactExtractor) would inherit
# from this class to ensure a consistent interface, for example, requiring an `execute` method.
# For now, it serves as a structural placeholder.
# ==================================================================================================

class BaseModule:
    """
    Abstract Base Class for all EnkiBot modules.
    """
    def __init__(self, name: str):
        self.name = name

    def execute(self, *args, **kwargs):
        """
        The main method to be implemented by all subclasses.
        """
        raise NotImplementedError("Each module must implement the 'execute' method.")
