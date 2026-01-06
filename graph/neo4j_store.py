"""Neo4j graph database operations."""
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase
from loguru import logger
from config.settings import settings


class Neo4jStore:
    """
    Manages Neo4j graph database operations.
    Stores entities (Document, Policy, Section, Topic, Concept) and relationships.
    """
    
    def __init__(self):
        """Initialize Neo4j connection."""
        self.logger = logger.bind(name=self.__class__.__name__)
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._ensure_constraints()
    
    @staticmethod
    def _is_primitive(value: Any) -> bool:
        """Check if a value is a primitive type (string, int, float, bool) or list of primitives."""
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, list):
            return all(Neo4jStore._is_primitive(item) for item in value)
        return False
    
    @staticmethod
    def _flatten_metadata(metadata: Dict) -> Dict[str, Any]:
        """
        Flatten metadata dictionary to only include primitive values.
        Neo4j only accepts primitive types (string, int, float, bool) or arrays of primitives.
        """
        if not isinstance(metadata, dict):
            return {}
        
        flattened = {}
        for key, value in metadata.items():
            # Skip non-primitive values (dicts, objects)
            if Neo4jStore._is_primitive(value):
                flattened[key] = value
            # Convert lists with mixed types to lists of strings
            elif isinstance(value, list):
                flattened[key] = [str(item) for item in value]
        
        return flattened
    
    def _ensure_constraints(self):
        """Create unique constraints for nodes."""
        with self.driver.session() as session:
            # Create constraints for unique node IDs
            constraints = [
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.debug(f"Constraint might already exist: {e}")
        
        self.logger.info("Neo4j constraints ensured")
    
    def create_document_node(self, doc_id: str, filename: str, metadata: Dict) -> str:
        """Create a Document node."""
        # Flatten metadata to individual properties (Neo4j doesn't support dict properties)
        flattened = self._flatten_metadata(metadata)
        
        with self.driver.session() as session:
            # Build SET clause dynamically for flattened metadata fields
            set_clauses = ["d.filename = $filename", "d.created_at = datetime()"]
            params = {"doc_id": doc_id, "filename": filename}
            
            # Add flattened metadata fields as individual properties
            for key, value in flattened.items():
                # Use a safe parameter name (replace dots/colons with underscores)
                param_key = f"meta_{key}".replace(".", "_").replace(":", "_")
                set_clauses.append(f"d.{key} = ${param_key}")
                params[param_key] = value
            
            query = f"""
            MERGE (d:Document {{id: $doc_id}})
            SET {', '.join(set_clauses)}
            RETURN d.id as id
            """
            params["doc_id"] = doc_id
            result = session.run(query, **params)
            return result.single()["id"]
    
    def create_chunk_node(self, chunk_id: str, text: str, metadata: Dict, doc_id: str):
        """Create a Chunk node and link it to Document."""
        # Flatten metadata to individual properties (Neo4j doesn't support dict properties)
        flattened = self._flatten_metadata(metadata)
        
        with self.driver.session() as session:
            # Build SET clause dynamically for flattened metadata fields
            set_clauses = ["ch.text = $text"]
            params = {
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": doc_id,
            }
            
            # Add chunk_index if available
            chunk_index = metadata.get("chunk_index", 0)
            params["chunk_index"] = chunk_index
            set_clauses.append("ch.index = $chunk_index")
            
            # Add flattened metadata fields as individual properties
            for key, value in flattened.items():
                # Skip chunk_id, text, and chunk_index as they're handled separately
                if key in ("chunk_id", "text", "chunk_index"):
                    continue
                # Use a safe parameter name
                param_key = f"meta_{key}".replace(".", "_").replace(":", "_")
                set_clauses.append(f"ch.{key} = ${param_key}")
                params[param_key] = value
            
            query = f"""
            MERGE (ch:Chunk {{id: $chunk_id}})
            SET {', '.join(set_clauses)}
            WITH ch
            MATCH (d:Document {{id: $doc_id}})
            MERGE (ch)-[:BELONGS_TO]->(d)
            RETURN ch.id as id
            """
            result = session.run(query, **params)
            return result.single()["id"]
    
    def create_entity_nodes(self, entities: List[Dict], chunk_id: str):
        """
        Create entity nodes (Policy, Section, Topic, Concept) and link to chunk.
        
        Args:
            entities: List of entity dictionaries with 'type', 'name', 'id', 'properties'
            chunk_id: ID of the chunk these entities belong to
        """
        with self.driver.session() as session:
            for entity in entities:
                entity_type = entity.get("type")
                entity_id = entity.get("id")
                entity_name = entity.get("name", "")
                properties = entity.get("properties", {})
                
                if not entity_type or not entity_id:
                    continue
                
                # Flatten properties to individual fields (Neo4j doesn't support dict properties)
                flattened_props = self._flatten_metadata(properties)
                
                # Build SET clause dynamically
                set_clauses = ["e.name = $name"]
                params = {
                    "entity_id": entity_id,
                    "name": entity_name,
                    "chunk_id": chunk_id,
                }
                
                # Add flattened properties as individual fields
                for key, value in flattened_props.items():
                    param_key = f"prop_{key}".replace(".", "_").replace(":", "_")
                    set_clauses.append(f"e.{key} = ${param_key}")
                    params[param_key] = value
                
                query = f"""
                MERGE (e:{entity_type} {{id: $entity_id}})
                SET {', '.join(set_clauses)}
                WITH e
                MATCH (ch:Chunk {{id: $chunk_id}})
                MERGE (e)-[:MENTIONED_IN]->(ch)
                RETURN e.id as id
                """
                
                session.run(query, **params)
        
        self.logger.info(f"Created {len(entities)} entity nodes for chunk {chunk_id}")
    
    def create_relationships(self, relationships: List[Dict]):
        """
        Create relationships between entities.
        
        Args:
            relationships: List of relationship dicts with 'from', 'to', 'type', 'properties'
        """
        with self.driver.session() as session:
            for rel in relationships:
                from_id = rel.get("from")
                to_id = rel.get("to")
                rel_type = rel.get("type", "RELATES_TO")
                properties = rel.get("properties", {})
                
                if not from_id or not to_id:
                    continue
                
                # Flatten properties to individual fields (Neo4j doesn't support dict properties)
                flattened_props = self._flatten_metadata(properties)
                
                # Build SET clause dynamically (only if there are properties)
                params = {
                    "from_id": from_id,
                    "to_id": to_id,
                }
                
                if flattened_props:
                    set_clauses = []
                    for key, value in flattened_props.items():
                        param_key = f"prop_{key}".replace(".", "_").replace(":", "_")
                        set_clauses.append(f"r.{key} = ${param_key}")
                        params[param_key] = value
                    
                    query = f"""
                    MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET {', '.join(set_clauses)}
                    RETURN r
                    """
                else:
                    # No properties, just create the relationship
                    query = f"""
                    MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    RETURN r
                    """
                
                session.run(query, **params)
        
        self.logger.info(f"Created {len(relationships)} relationships")
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """Retrieve chunks by their IDs."""
        with self.driver.session() as session:
            # Get all properties and reconstruct metadata dict (metadata was flattened to individual properties)
            query = """
            MATCH (ch:Chunk)
            WHERE ch.id IN $chunk_ids
            OPTIONAL MATCH (ch)-[:BELONGS_TO]->(d:Document)
            RETURN ch.id as id, ch.text as text, properties(ch) as props,
                   d.filename as document_filename
            """
            result = session.run(query, chunk_ids=chunk_ids)
            
            # Reconstruct metadata dict from properties (excluding core fields)
            chunks = []
            for record in result:
                props = dict(record["props"])
                # Core fields
                chunk_id = props.pop("id", record["id"])
                text = props.pop("text", record["text"])
                index = props.pop("index", None)
                
                # Everything else is metadata
                metadata = props.copy()
                if index is not None:
                    metadata["chunk_index"] = index
                
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "document_filename": record["document_filename"],
                })
            
            return chunks
    
    def traverse_from_chunks(self, chunk_ids: List[str], max_hops: int = 2) -> List[Dict]:
        """
        Traverse graph from initial chunks to find related context.
        This is the graph traversal part of GraphRAG.
        
        Args:
            chunk_ids: Initial chunk IDs from vector search
            max_hops: Maximum number of relationship hops
            
        Returns:
            List of related nodes with their types and relationships
        """
        with self.driver.session() as session:
            # Get all properties and reconstruct metadata dict (metadata was flattened to individual properties)
            query = f"""
            MATCH path = (ch:Chunk)-[*1..{max_hops}]-(related)
            WHERE ch.id IN $chunk_ids
            WITH DISTINCT related, ch
            RETURN related.id as id,
                   labels(related) as labels,
                   related.name as name,
                   related.text as text,
                   properties(related) as props,
                   ch.id as source_chunk_id
            LIMIT 100
            """
            result = session.run(query, chunk_ids=chunk_ids)
            
            # Reconstruct metadata dict from properties
            nodes = []
            for record in result:
                props = dict(record["props"])
                # Core fields
                node_id = props.pop("id", record["id"])
                name = props.pop("name", record.get("name"))
                text = props.pop("text", record.get("text"))
                
                # Everything else is metadata
                metadata = props.copy()
                
                nodes.append({
                    "id": node_id,
                    "labels": record["labels"],
                    "name": name,
                    "text": text,
                    "metadata": metadata,
                    "source_chunk_id": record["source_chunk_id"],
                })
            
            return nodes
    
    def clear_all(self):
        """Clear all nodes and relationships (for testing/reset)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.logger.warning("Cleared all nodes and relationships from Neo4j")
    
    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()
        self.logger.info("Neo4j connection closed")





