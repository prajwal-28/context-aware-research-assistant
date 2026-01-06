"""Entity and relationship extraction from text chunks."""
from typing import List, Dict
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from loguru import logger
from config.settings import settings
import json


class EntityExtractor:
    """
    Extracts entities (Policy, Section, Topic, Concept) and relationships from text.
    Uses LLM to identify semantic entities and their connections.
    """
    
    ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from documents.

Given the following text chunk, extract entities and relationships.

Entity Types:
- Document: The overall document (already known)
- Policy: Policies, rules, regulations mentioned
- Section: Document sections, chapters, parts
- Topic: Main topics or themes discussed
- Concept: Important concepts, ideas, terms

For each entity, provide:
- type: One of Policy, Section, Topic, Concept
- id: A unique identifier (e.g., "policy_maternity_leave", "topic_project_management")
- name: A short descriptive name
- properties: Any relevant metadata (optional)

For relationships, provide:
- from: Source entity ID
- to: Target entity ID
- type: Relationship type (e.g., "AFFECTS", "RELATES_TO", "CONTAINS", "REFERENCES")
- properties: Any relevant metadata (optional)

Text chunk:
{text}

Return ONLY a valid JSON object with this structure:
{{
    "entities": [
        {{"type": "Policy", "id": "...", "name": "...", "properties": {{}}}},
        ...
    ],
    "relationships": [
        {{"from": "entity_id_1", "to": "entity_id_2", "type": "AFFECTS", "properties": {{}}}},
        ...
    ]
}}"""

    def __init__(self):
        """Initialize entity extractor with LLM."""
        self.logger = logger.bind(name=self.__class__.__name__)
        try:
            self.llm = OpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o-mini",  # Using mini for cost efficiency
                temperature=0.0,  # Deterministic extraction
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def extract(self, text: str, chunk_id: str) -> tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Text chunk to extract from
            chunk_id: ID of the chunk (for entity IDs)
            
        Returns:
            Tuple of (entities list, relationships list)
        """
        response_text = ""
        try:
            self.logger.debug(f"Extracting entities from chunk {chunk_id}")
            
            prompt = PromptTemplate(self.ENTITY_EXTRACTION_PROMPT)
            formatted_prompt = prompt.format(text=text[:4000])  # Limit text length
            
            response = self.llm.complete(formatted_prompt)
            response_text = str(response).strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
            
            # Prefix entity IDs with chunk_id to ensure uniqueness
            for entity in entities:
                if "id" in entity:
                    entity["id"] = f"{chunk_id}_{entity['id']}"
            
            # Update relationship IDs
            entity_id_map = {e.get("id", ""): e.get("id", "") for e in entities}
            for rel in relationships:
                from_id = rel.get("from", "")
                to_id = rel.get("to", "")
                if from_id in entity_id_map:
                    rel["from"] = entity_id_map[from_id]
                else:
                    # If entity not in this chunk, try to find it
                    rel["from"] = f"{chunk_id}_{from_id}"
                if to_id in entity_id_map:
                    rel["to"] = entity_id_map[to_id]
                else:
                    rel["to"] = f"{chunk_id}_{to_id}"
            
            self.logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            return entities, relationships
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}. Response: {response_text[:200] if response_text else 'No response'}")
            return [], []
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return [], []

