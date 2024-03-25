"""
© 2024 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""
from typing import Type, List

from genai_core.actors.base import ActorInput, ActorOutput
from genai_core.actors.open_ai_actor import OpenAIActor
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage,SystemMessage
from pydantic import BaseModel, Field

from genai_chain_sql.actors.conversation_actor import ConversationActor
from genai_chain_sql.constants import *
from genai_chain_sql.actors.examples.metadata_actor_examples import _V4_METADATA_ACTOR_EXAMPLES

import json

# Me está eliminando viga artesa de la pregunta compuesta, incluso siendo una colujna de tabla a utilizar
# 1. Dismiss tables that are not essential to construct the SQL QUERY to address the USER REQUEST without taking into consideration whether contains sensitive data or not.


INSTRUCTIONS_v3 = f"""Act as an Business Expert on data domains.
You will receive a DATA DOMAIN, which consists of a collection of database tables, each detailed with a business description and their respective columns, alongside a USER REQUEST. 
Your task involves a thorough examination and comprehension of both the USER REQUEST and the details encapsulated within the DATA DOMAIN, basing your analysis strictly on the context provided without resorting to any external information or assumptions not explicitly mentioned.
Your objective is:
1. To find all group of words (mix of words, codes, symbols and figures) which might contain mispellings or typos and together make a chunk from the USER REQUEST that are closely related to any column that would be use within the WHERE clause of a SQL QUERY.
2. To find whether this column's table contains a description which explicitly states this column to contain sensitive info within the DATA DOMAIN. 
3. Come up with a general word for the group of words previously found.

All your findings, irrespective of containing sensitive info or not, should be organized into a list of JSON objects, formatted as follows:
-chunk: The group of words found.
-name: The name of the column that should be used along with the group of words to construct parts of a WHERE clause in a SQL QUERY.
-sensitive: A boolean indicating whether this column has sensitive data.
-chunk_reason: A concise justification for the chunks's inclusion or exclusion.
-sensitive_reason: A concise justification for the column's sensitive information presence of absence.
-generalization: Choose a word that generalizes the previous found chunk.
-generalization_reason: A concise justification for the chosen generalization word.

Ensure that the USER REQUEST can be adequately addressed using the tables deemed necessary.

The DATA DOMAIN is defined in the following format:
```
{GOVERNANCE_TABLE_FORMAT}
```
"""

# INSTRUCTIONS_v3 = f"""Act as an Business Expert on data domains.
# You will receive a DATA DOMAIN, which consists of a collection of database tables, each detailed with a business description and their respective columns, alongside a USER REQUEST. 
# Your task involves a thorough examination and comprehension of both the USER REQUEST and the details encapsulated within the DATA DOMAIN, basing your analysis strictly on the context provided without resorting to any external information or assumptions not explicitly mentioned.
# Your objective is:
# 1. To find all group of words (mix of words, codes, symbols and figures) which might contain mispellings or typos and together make a chunk from the USER REQUEST that are closely related to any column that would be use within the WHERE clause of a SQL QUERY.
# 2. To find whether this column's table contains a description which explicitly states this column to contain sensitive info within the DATA DOMAIN. 

# All your findings, irrespective of containing sensitive info or not, should be organized into a list of JSON objects, formatted as follows:
# -chunk: The group of words found.
# -name: The name of the column that should be used along with the group of words to construct parts of a WHERE clause in a SQL QUERY.
# -sensitive: A boolean indicating whether this column has sensitive data.
# -chunk_reason: A concise justification for the chunks's inclusion or exclusion.
# -sensitive_reason: A concise justification for the column's sensitive information presence of absence.
# Ensure that the USER REQUEST can be adequately addressed using the tables deemed necessary.

# The DATA DOMAIN is defined in the following format:
# ```
# {GOVERNANCE_TABLE_FORMAT}
# ```
# """


INPUT_TEMPLATE = f"""DATA DOMAIN:
###
{{{COLLECTION_TABLES_CONTEXT_KEY}}}
###

USER REQUEST:
###
{{{ConversationActor.actor_key}}}
###
"""

EXAMPLE_CONTEXT = """[Table(learning.customer, description:'Almacena información de los clientes, incluyendo datos personales y de contacto. La columna name almacena informacion sensible', columns:[('customer_id', INTEGER), ('name', TEXT)]) 
Table(learning.product, description:'Contiene detalles sobre los productos ofrecidos, incluyendo precios y categorías.', columns:[('product_id', TEXT), ('name', TEXT), ('price', DOUBLE)]) 
"""

class Extraction(BaseModel):
    chunk: str = Field()
    name: str = Field()
    sensitive: bool = Field()
    chunk_reason: str = Field()
    sensitive_reason: str = Field()
    generalization: str = Field()
    generalization_reason: str = Field()


class MetadataActorOutputModel(BaseModel):
    extraction_list: List[Extraction] = Field()



class MetadataActorOutput(ActorOutput):
    @classmethod
    def get_pydantic_object(cls) -> Type[BaseModel]:
        return MetadataActorOutputModel


class MetadataActorInput(ActorInput):
    template = INPUT_TEMPLATE
    _input_variables = [COLLECTION_TABLES_CONTEXT_KEY, ConversationActor.actor_key]


class MetadataActor(OpenAIActor):
    actor_key = "metadata_actor"
    temperature = 0.2
    model_name = OPENAI_GPT3_MODEL
    retry_model_name = model_name

    METADATA_ACTOR_EXAMPLES: List[dict] = _V4_METADATA_ACTOR_EXAMPLES    

    @classmethod
    def input_type(cls) -> Type[ActorInput]:
        return MetadataActorInput

    @classmethod
    def output_type(cls) -> Type[ActorOutput]:
        return MetadataActorOutput

    @classmethod
    def instructions(cls) -> str:
        return INSTRUCTIONS_v3

    @classmethod
    def examples(cls) -> List[BaseMessage]:

        examples_list = []
        for example in cls.METADATA_ACTOR_EXAMPLES:
            complete_example = [
                HumanMessage(
                    content=MetadataActorInput.create_input(
                        **{COLLECTION_TABLES_CONTEXT_KEY: example[COLLECTION_TABLES_CONTEXT_KEY],
                           ConversationActor.actor_key: example["query"]})
                ),
                AIMessage(
                    content=json.dumps(example["query_metadata"])
                )
            ]

            examples_list += complete_example

        return examples_list         


    @classmethod
    def format_output(cls):
        return lambda x: x.dict()
