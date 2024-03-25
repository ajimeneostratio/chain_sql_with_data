"""
© 2024 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from genai_chain_sql.actors.examples.sql_actor_examples import _ADIF_NO_DATA_SQL_ACTOR_EXAMPLES, _ADIF_WITH_DATA_SQL_ACTOR_EXAMPLES
from genai_chain_sql.actors.explain_actor import ExplainActor
from genai_chain_sql.actors.metadata_actor import MetadataActor
from genai_chain_sql.actors.precision_actor import PrecisionActor
from genai_chain_sql.actors.sql_actor import EXAMPLE_INJECTION, SQLActor, SQLActorInput
from genai_chain_sql.helpers.chain_logger import ChainLogger
import pytest
from langchain_core.runnables import RunnableConfig

from genai_chain_sql.constants import *
from genai_chain_sql.sql_chain_hybrid import SQLChain
from genai_core.vectorstores.opensearch_ml_vector_search import OpenSearchMlVectorSearch
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage
from genai_chain_sql.nlp_processing import *
from genai_core.clients.governance.governance_client import (
    DataAsset,
    GovernanceGlossaryItemsResponse,
    GovernanceClient,
    GlossaryItem
)
from genai_core.services.governance.model.governance_data_asset import (
    GovernanceDataAsset,
    GovernanceColumn,
    GovernanceBusinessAsset,
)


from typing import Dict, Optional
import os
import time
import unidecode
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid4

from bs4 import BeautifulSoup
from genai_core.chain.base import BaseGenAiChain
from genai_core.clients.vault.vault_client import VaultClient
from genai_core.constants.constants import (
    ENV_VAR_GENAI_API_TENANT,
    ENV_VAR_GENAI_API_SERVICE_NAME,
)
from genai_core.logger.logger import log
from genai_core.memory.dict_memory import DictMemory
from genai_core.runnables.genai_auth import GenAiAuthRunnable
from genai_core.services.governance.governance_service import GovernanceService
from genai_core.vectorstores.opensearch_ml_vector_search import OpenSearchMlVectorSearch
from genai_core.services.governance.model.governance_data_asset import (
    GovernanceDataAsset,
)
from genai_core.services.virtualizer.virtualizer_service import VirtualizerService
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda,
    RunnableConfig,
)

from genai_chain_sql.actors.context_actor import ContextActor, InformationQuality
from genai_chain_sql.actors.conversation_actor import ConversationActor
from genai_chain_sql.actors.metadata_actor import MetadataActor
from genai_chain_sql.actors.tables_actor import (
    TablesActor,
    TablesActorOutputModel,
)
from genai_chain_sql.constants import *
from genai_chain_sql.graph.sql_graph import LOG_PREFIX, SQLGraph
from genai_chain_sql.helpers.chain_helpers import log_chain, extract_uid
from genai_chain_sql.helpers.chain_logger import ChainLogger



class GenericChain(BaseGenAiChain):

    # Services
    governance_service: GovernanceService
    virtualizer_service: VirtualizerService
    opensearch_vectorstore: OpenSearchMlVectorSearch

    # Caches
    governance_collection_cache: Dict[
        str, Tuple[int, List[GovernanceDataAsset]]
    ] = dict()
    governance_cache_ttl_min: int
    chat_memory: DictMemory = DictMemory(memory_key=INPUT_KEY, save_output=False)
    chat_memory_id_cache: Dict[str, int] = dict()
    chat_cache_ttl_min: int

    openai_api_key: str
    llm_timeout : int = 30

    @classmethod
    def __init__(
        cls,
        virtualizer_host: str = "127.0.0.1",
        virtualizer_port: int = 13422,
        governance_url: str = "https://127.0.0.1:60000",
        opensearch_url: str = "opensearch-node:9200",
        opensearch_index: str = "adif-materials-nlp-index",
        openai_stratio_credential: str = "openai-token",
        opensearch_embedding_model_id: Optional[str] = "PzOM-Y0BQbWlGWXIt3dp",        
        virtualizer_timeout: int = 60,
        governance_timeout: int = 10,
        graph_max_retries: int = 3,
        governance_cache_ttl_min: int = 30,
        chat_cache_ttl_min: int = 120,
        llm_timeout: int = 30,
    ):
        # Check required params
        missing_params = [
            param_name
            for param_name, param_value in [
                ("virtualizer_host", virtualizer_host),
                ("virtualizer_port", virtualizer_port),
                ("governance_url", governance_url),
                ("openai_stratio_credential", openai_stratio_credential),
                ("opensearch_url", opensearch_url),
                ("opensearch_index", opensearch_index),
                ("opensearch_embedding_model_id", opensearch_embedding_model_id),
            ]
            if param_value is None
        ]
        if missing_params:
            raise ValueError(f"Required value(s) missing: {', '.join(missing_params)}")

        # Init vault and extract secrets
        vault_client = VaultClient()
        secret = vault_client.get_password(openai_stratio_credential)
        if "token" not in secret:
            raise ValueError(
                f"The secret in Vault is malformed. It has to contain the 'token' field."
            )
        cls.openai_api_key = secret["token"]
        (client_cert, client_key) = vault_client.get_service_certificate_pem_paths()
        ca_file = vault_client.get_ca_bundle_pem_path()

        # Init services
        tenant = os.environ.get(ENV_VAR_GENAI_API_TENANT)
        service_name = os.environ.get(ENV_VAR_GENAI_API_SERVICE_NAME)
        cls.governance_service = GovernanceService(
            url=governance_url,
            ca_certs=ca_file,
            client_cert=client_cert,
            client_key=client_key,
            headers_tenant=tenant,
            headers_user=service_name,
            headers_role="Service",
            semantic_prefix="semantic_",
            max_governance_threads=10,
            request_timeout=governance_timeout,
        )
        cls.virtualizer_service = VirtualizerService(
            host=virtualizer_host,
            port=virtualizer_port,
            username=service_name,
            ca_certs=ca_file,
            client_cert=client_cert,
            client_key=client_key,
            request_timeout=virtualizer_timeout,
        )

        cls.opensearch_vectorstore = OpenSearchMlVectorSearch(
            opensearch_url=opensearch_url,
            index_name=opensearch_index,
            embedding_model_id=opensearch_embedding_model_id,
            http_auth=("admin", "admin"),
            use_ssl = True,
            verify_certs = False,
            ssl_assert_hostname = False,
            ssl_show_warn = False)

        # Init other chain resources
        cls.governance_cache_ttl_min = governance_cache_ttl_min * 60
        cls.chat_cache_ttl_min = chat_cache_ttl_min * 60

        log.info("Generic Chain ready!")



    @classmethod
    def retrieve_collection_information(cls, data: Dict[str, Any]) -> str:
        # TODO: we should count tokens on this method and limit to avoid runtime errors
        governance_collection_info = cls.get_collection_documents_from_cache(
            data[COLLECTION_INPUT_KEY]
        )
        governance_allowed_info = cls.get_user_allowed_tables(
            data, governance_collection_info
        )

        # return cls.compress_tables(governance_allowed_info, extended=False) 
        # return "Table(semantic_BDP.Catalogo, 'El importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa tabla Catalogo contiene información de la relación que hay entre los distintos niveles jerárquicos del catálogo de la base de precios. Cada entrada en la tabla representa un concepto de la base de precios con su nivel de agregación inferior y superior.La tabla Catalogo está relacionada con la tabla Recursos. Esta relación se establece cuando la columna codigoNivelInferior de la tabla Catalogo y la columna codigoRecurso de la tabla Recursos tienen el mismo valor.El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelinferior de la tabla Catalogo.', columns: '[(descripcionCatalogo, TEXT), (codigoNivelInferior, TEXT), (codigoCatalogo, TEXT), (codigoNivelSuperior, TEXT), (cantidad, DECIMAL), (categoria, TEXT)]')\nTable(semantic_BDP.Recursos, ' \nEl importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa jornada es una unidad de medida relativa al tiempo. 1 jornada es igual a 8 horas.Por ejemplo, para calcular el precio de jornadas, si valor unidadRecurso es 'h' y precioRecurso es 10, el precio en jornadas será 10*8=80.Por ejemplo, para calcular la cantidad de jornadas, si valor unidadRecurso es 'h' y cantidad es 16, la cantidad en jornadas será 16/8=2.\nLa tabla Recursos contiene información de los recursos de la base de precios. Cada entrada en la tabla representa un recurso específico.Los recursos también pueden ser llamados 'elementos', 'medios' o 'componentes'.La tabla Recursos está relacionada con la tabla Catalogo. Esta relación se establece cuando la columna codigoRecurso de la tabla Recursos y la columna codigoNivelInferior de la tabla Catalogo tienen el mismo valor. El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelInferior de la tabla Catalogo.Para saber que puedo construir con un recurso, habrá que filtrar el recurso en la columna descripcionRecurso de la tabla Recursos y ver su relación con la tabla Catalogo para obtener la columna descripcionCatalogo.Siempre que se pida precio, tipo y unidad, se tiene que consultar la tabla Recursos.', columns: '[(codigoRecurso, TEXT), (tipoRecurso, TEXT), (unidadRecurso, TEXT), (precioRecurso, DECIMAL), (descripcionRecurso, TEXT)]')\nTable(semantic_BDP.Jerarquias, columns: '[(precio, DECIMAL), (codigoRecurso, TEXT), (unidad, TEXT), (codigo, TEXT), (categoria, TEXT), (recurso, TEXT)]')"
        return "Table(semantic_BDP.Catalogo, 'El importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa tabla Catalogo contiene información de la relación que hay entre los distintos niveles jerárquicos del catálogo de la base de precios. Cada entrada en la tabla representa un concepto de la base de precios con su nivel de agregación inferior y superior.La tabla Catalogo está relacionada con la tabla Recursos. Esta relación se establece cuando la columna codigoNivelInferior de la tabla Catalogo y la columna codigoRecurso de la tabla Recursos tienen el mismo valor.El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelinferior de la tabla Catalogo.', columns: '[(descripcionCatalogo, TEXT), (codigoNivelInferior, TEXT), (codigoCatalogo, TEXT), (codigoNivelSuperior, TEXT), (cantidad, DECIMAL), (categoria, TEXT)]')\nTable(semantic_BDP.Recursos, ' \nEl importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa jornada es una unidad de medida relativa al tiempo. 1 jornada es igual a 8 horas.Por ejemplo, para calcular el precio de jornadas, si valor unidadRecurso es 'h' y precioRecurso es 10, el precio en jornadas será 10*8=80.Por ejemplo, para calcular la cantidad de jornadas, si valor unidadRecurso es 'h' y cantidad es 16, la cantidad en jornadas será 16/8=2.\nLa tabla Recursos contiene información de los recursos de la base de precios. Cada entrada en la tabla representa un recurso específico.Los recursos también pueden ser llamados 'elementos', 'medios' o 'componentes'.La tabla Recursos está relacionada con la tabla Catalogo. Esta relación se establece cuando la columna codigoRecurso de la tabla Recursos y la columna codigoNivelInferior de la tabla Catalogo tienen el mismo valor. El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelInferior de la tabla Catalogo.Para saber que puedo construir con un recurso, habrá que filtrar el recurso en la columna descripcionRecurso de la tabla Recursos y ver su relación con la tabla Catalogo para obtener la columna descripcionCatalogo.', columns: '[(codigoRecurso, TEXT), (tipoRecurso, TEXT), (unidadRecurso, TEXT), (precioRecurso, DECIMAL), (descripcionRecurso, TEXT)]')\nTable(semantic_BDP.Jerarquias, columns: '[(precio, DECIMAL), (codigoRecurso, TEXT), (unidad, TEXT), (codigo, TEXT), (categoria, TEXT), (recurso, TEXT)]')"

    @classmethod
    def retrieve_tables_information(cls, data: Dict[str, Any]) -> str:
        # TODO: we should count tokens on this method and limit to avoid runtime errors
        governance_collection_info = cls.get_collection_documents_from_cache(
            data[COLLECTION_INPUT_KEY]
        )
        requested_tables = data.get(REQUIRED_TABLES_KEY)
        governance_tables_info = [
            t for t in governance_collection_info if t.name in requested_tables
        ]
        governance_allowed_info = cls.get_user_allowed_tables(
            data, governance_tables_info
        )
        # return cls.compress_tables(governance_allowed_info, extended=True)    
        # return "Table(semantic_BDP.Catalogo, 'El importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa tabla Catalogo contiene información de la relación que hay entre los distintos niveles jerárquicos del catálogo de la base de precios. Cada entrada en la tabla representa un concepto de la base de precios con su nivel de agregación inferior y superior.La tabla Catalogo está relacionada con la tabla Recursos. Esta relación se establece cuando la columna codigoNivelInferior de la tabla Catalogo y la columna codigoRecurso de la tabla Recursos tienen el mismo valor.El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelinferior de la tabla Catalogo.', columns: '[(Catalogo.descripcionCatalogo, TEXT, 'La columna descripcionCatalogo, contiene la descripción de un concepto del catálogo.'), (Catalogo.codigoNivelInferior, TEXT, 'La columna codigoNivelInferior, contiene el identificador del elemento del catálogo que pertenece a descripcionCatalogo, es decir, que está a un nivel inferior en la jerarquía del catálogo.'), (Catalogo.codigoCatalogo, TEXT, 'La columna codigoCatalogo, contiene el identificador de cada elemento del catálogo.'), (Catalogo.codigoNivelSuperior, TEXT, 'La columna codigoNivelSuperior, contiene el identificador del elemento del catálogo al que pertenece descripcionCatalogo, es decir, que está a un nivel superior en la jerarquía del catálogo.'), (Catalogo.cantidad, DECIMAL, 'La columna cantidad, contiene la cantidad del elemento cuyo identificador está en la columna codigoNivelInferior que es necesario para construir o fabricar el elemento del catálogo del campo descripcionCatalogo.La cantidad debe mostrarse siempre con 4 decimales: ROUND(cantidad,4) AS cantidad'), (Catalogo.categoria, TEXT, 'La columna categoria, contiene la categoria del catálogo a la que pertenece descripcionCatalogo, es decir, que está a un nivel superior en la jerarquía del catálogo.')]')\nTable(semantic_BDP.Recursos, ' \nEl importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa jornada es una unidad de medida relativa al tiempo. 1 jornada es igual a 8 horas.Por ejemplo, para calcular el precio de jornadas, si valor unidadRecurso es 'h' y precioRecurso es 10, el precio en jornadas será 10*8=80.Por ejemplo, para calcular la cantidad de jornadas, si valor unidadRecurso es 'h' y cantidad es 16, la cantidad en jornadas será 16/8=2.\nLa tabla Recursos contiene información de los recursos de la base de precios. Cada entrada en la tabla representa un recurso específico.Los recursos también pueden ser llamados 'elementos', 'medios' o 'componentes'.La tabla Recursos está relacionada con la tabla Catalogo. Esta relación se establece cuando la columna codigoRecurso de la tabla Recursos y la columna codigoNivelInferior de la tabla Catalogo tienen el mismo valor. El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelInferior de la tabla Catalogo.Para saber que puedo construir con un recurso, habrá que filtrar el recurso en la columna descripcionRecurso de la tabla Recursos y ver su relación con la tabla Catalogo para obtener la columna descripcionCatalogo.Siempre que se pida precio, tipo y unidad, se tiene que consultar la tabla Recursos.', columns: '[(Recursos.codigoRecurso, TEXT, 'La columna codigoRecurso, contiene el identificador de cada recurso.'), (Recursos.tipoRecurso, TEXT, 'Los posibles valores son:'MAQUINARIA'. Podemos referirnos a los recursos de tipo MAQUINARIA cuando hablamos de maquinas, mecanismos...'PERSONAL'. Podemos referirnos a los recursos de tipo PERSONAL cuando hablamos de personas, profesionales, trabajadores, expertos, constructores...'MATERIAL'. Podemos referirnos a los recursos de tipo MATERIAL cuando hablamos de utilitarios, materiales, herramientas, equipos, utensilios, equipamiento...'), (Recursos.unidadRecurso, TEXT, 'La columna unidadRecurso, contiene la unidad de medida de cada recurso. '), (Recursos.precioRecurso, DECIMAL, 'La columna precioRecurso, contiene el precio de cada recurso.El precio debe mostrarse siempre con 2 decimales de la siguiente forma ROUND(Recursos.precioRecurso,2) AS precio'), (Recursos.descripcionRecurso, TEXT, 'La columna descripcionRecurso, contiene la descripción de cada recurso. Cuando queremos saber información de los recursos filtramos por esta columna.')]')"   
        return "Table(semantic_BDP.Catalogo, 'El importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa tabla Catalogo contiene información de la relación que hay entre los distintos niveles jerárquicos del catálogo de la base de precios. Cada entrada en la tabla representa un concepto de la base de precios con su nivel de agregación inferior y superior.La tabla Catalogo está relacionada con la tabla Recursos. Esta relación se establece cuando la columna codigoNivelInferior de la tabla Catalogo y la columna codigoRecurso de la tabla Recursos tienen el mismo valor.El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelinferior de la tabla Catalogo.', columns: '[(Catalogo.descripcionCatalogo, TEXT, 'La columna descripcionCatalogo, contiene la descripción de un concepto del catálogo.'), (Catalogo.codigoNivelInferior, TEXT, 'La columna codigoNivelInferior, contiene el identificador del elemento del catálogo que pertenece a descripcionCatalogo, es decir, que está a un nivel inferior en la jerarquía del catálogo.'), (Catalogo.codigoCatalogo, TEXT, 'La columna codigoCatalogo, contiene el identificador de cada elemento del catálogo.'), (Catalogo.codigoNivelSuperior, TEXT, 'La columna codigoNivelSuperior, contiene el identificador del elemento del catálogo al que pertenece descripcionCatalogo, es decir, que está a un nivel superior en la jerarquía del catálogo.'), (Catalogo.cantidad, DECIMAL, 'La columna cantidad, contiene la cantidad del elemento cuyo identificador está en la columna codigoNivelInferior que es necesario para construir o fabricar el elemento del catálogo del campo descripcionCatalogo.La cantidad debe mostrarse siempre con 4 decimales: ROUND(cantidad,4) AS cantidad'), (Catalogo.categoria, TEXT, 'La columna categoria, contiene la categoria del catálogo a la que pertenece descripcionCatalogo, es decir, que está a un nivel superior en la jerarquía del catálogo.')]')\nTable(semantic_BDP.Recursos, ' \nEl importe es el monto, cuantía o coste que habrá que pagar por un recurso necesario para hacer una construcción.El importe se calcula multiplicando el campo cantidad de la tabla Catalogo por el campo precioRecurso de la tabla Recurso.Por ejemplo, si el precio del recurso 'AYUDANTE' es 10 y la cantidad necesaria para construir 'ESTRUCTURAS' es 2, el importe será: 10*2=20.El importe debe mostrarse siempre con 2 decimales: ROUND((precioRecurso*cantidad),2) AS importe.\nLa jornada es una unidad de medida relativa al tiempo. 1 jornada es igual a 8 horas.Por ejemplo, para calcular el precio de jornadas, si valor unidadRecurso es 'h' y precioRecurso es 10, el precio en jornadas será 10*8=80.Por ejemplo, para calcular la cantidad de jornadas, si valor unidadRecurso es 'h' y cantidad es 16, la cantidad en jornadas será 16/8=2.\nLa tabla Recursos contiene información de los recursos de la base de precios. Cada entrada en la tabla representa un recurso específico.Los recursos también pueden ser llamados 'elementos', 'medios' o 'componentes'.La tabla Recursos está relacionada con la tabla Catalogo. Esta relación se establece cuando la columna codigoRecurso de la tabla Recursos y la columna codigoNivelInferior de la tabla Catalogo tienen el mismo valor. El valor del campo codigoRecurso de la tabla Recurso puede aparecer múltiples veces en el campo codigoNivelInferior de la tabla Catalogo.Para saber que puedo construir con un recurso, habrá que filtrar el recurso en la columna descripcionRecurso de la tabla Recursos y ver su relación con la tabla Catalogo para obtener la columna descripcionCatalogo.', columns: '[(Recursos.codigoRecurso, TEXT, 'La columna codigoRecurso, contiene el identificador de cada recurso.'), (Recursos.tipoRecurso, TEXT, 'Los posibles valores son:'MAQUINARIA'. Podemos referirnos a los recursos de tipo MAQUINARIA cuando hablamos de maquinas, mecanismos...'PERSONAL'. Podemos referirnos a los recursos de tipo PERSONAL cuando hablamos de personas, profesionales, trabajadores, expertos, constructores...'MATERIAL'. Podemos referirnos a los recursos de tipo MATERIAL cuando hablamos de utilitarios, materiales, herramientas, equipos, utensilios, equipamiento...'), (Recursos.unidadRecurso, TEXT, 'La columna unidadRecurso, contiene la unidad de medida de cada recurso. '), (Recursos.precioRecurso, DECIMAL, 'La columna precioRecurso, contiene el precio de cada recurso.El precio debe mostrarse siempre con 2 decimales de la siguiente forma ROUND(Recursos.precioRecurso,2) AS precio'), (Recursos.descripcionRecurso, TEXT, 'La columna descripcionRecurso, contiene la descripción de cada recurso. Cuando queremos saber información de los recursos filtramos por esta columna.')]')"


    @classmethod
    def get_collection_documents_from_cache(
        cls, collection_name: str
    ) -> List[GovernanceDataAsset]:
        current_time = int(time.time())
        cache_time, collection_data = cls.governance_collection_cache.get(
            collection_name, (0, [])
        )
        is_expired = (current_time - cache_time) > cls.governance_cache_ttl_min
        if is_expired:
            log.info(f"Expired collection cache '{collection_name}'")
            collection_data = cls.get_collection_table_documents(collection_name)
            cls.governance_collection_cache.update(
                {collection_name: (int(time.time()), collection_data)}
            )
            log.info(f"Updated collection cache '{collection_name}'")
        return cls.governance_collection_cache[collection_name][1]

    @classmethod
    def get_collection_table_documents(
        cls, collection_name: str
    ) -> List[GovernanceDataAsset]:
        data = cls.governance_service.get_governance_tables(collection_name)
        tables = [table.name for table in data]
        out = cls.governance_service.get_governance_collection_data(tables)
        return out

    @classmethod
    def get_examples_from_data_collection(
        cls, collection_name: str
    ) -> List[GovernanceDataAsset]:
        data = cls.governance_service.get_governance_tables(collection_name)
        tables = [table.name for table in data]
        out = cls.get_governance_examples_info(tables)
        return out

    @staticmethod
    def compress_tables(
        governance_info: List[GovernanceDataAsset], extended: bool
    ) -> str:
        table_outputs = []
        for table in governance_info:
            table_name = table.name
            table_description = BeautifulSoup(
                "\n".join(table.descriptions), "html.parser"
            ).get_text()
            if len(table_description) > 0:
                table_description = f", '{table_description}'"
            column_outputs = []
            for column in table.columns:
                column_name = column.name
                gov_type = column.type
                if extended:
                    only_table = table_name.split(".")[1]
                    column_name = f"{only_table}.{column_name}"
                    column_description = BeautifulSoup(
                        "\n".join([c.description for c in column.business_assets]),
                        "html.parser",
                    ).get_text()
                    if len(column_description) > 0:
                        column_description = f", '{column_description}'"
                else:
                    column_description = ""
                column_outputs.append(
                    f"({column_name}, {gov_type}{column_description})"
                )
            columns_str = ", ".join(column_outputs)
            table_outputs.append(
                f"Table({table_name}{table_description}, columns: '[{columns_str}]')"
            )
            print(f"Table({table_name}{table_description}, columns: '[{columns_str}]')")
        return "\n".join(table_outputs)

    @classmethod
    def get_user_allowed_tables(
        cls, data: Dict[str, Any], governance_info: List[GovernanceDataAsset]
    ) -> List[GovernanceDataAsset]:
        uid = extract_uid(data)
        for _ in range(0, 3):
            try:
                allowed_tables = [
                    f"{i['database']}.{i['tableName']}"
                    for i in cls.virtualizer_service.run_query(
                        f"show tables in {data[COLLECTION_INPUT_KEY]}", uid
                    ).result
                ]
                break
            except Exception:
                allowed_tables = []

        governance_allowed_info = []
        for table in governance_info:
            if table.name in allowed_tables:
                governance_allowed_info.append(table)
        return governance_allowed_info  
    
    @classmethod
    def filter_required_tables(cls, data: Dict[str, Any]) -> List[str]:
        tables_actor_output: TablesActorOutputModel = data.get(TablesActor.actor_key)
        required_tables: List[str] = []
        for table in tables_actor_output:
            if table.accepted:
                required_tables.append(table.name)
        return required_tables    
                 
    @staticmethod
    def get_or_create_chat_id(data: Dict[str, Any]) -> str:
        ChainLogger.info(
            f"Starting with question '{data.get(INPUT_KEY)}' on collection '{data.get(COLLECTION_INPUT_KEY)}'",
            data,
        )
        return data.get(CHAT_ID_KEY, str(uuid4()))
    
    @classmethod
    def update_chat_memory_cache(cls, chat_id: str):
        current_time = int(time.time())
        cls.chat_memory_id_cache.update({chat_id: current_time})
        for cache_id, cache_time in cls.chat_memory_id_cache.copy().items():
            is_expired = (current_time - cache_time) > cls.chat_cache_ttl_min
            if is_expired:
                log.info(f"Cleaning cache chat_memory id {cache_id}")
                cls.chat_memory_id_cache.pop(cache_id, None)
                cls.chat_memory.clear(cache_id)

    @staticmethod
    def get_user_chat_id(data: Dict[str, Any]) -> str:
        uid = extract_uid(data)
        chat_id = data.get(CHAT_ID_KEY)
        user_chat_id = f"{chat_id}{'-'+uid if uid is not None else ''}"
        return user_chat_id    
    @classmethod
    def load_chat_memory(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        ChainLogger.debug("Loading chat memory", data)
        chat_id = cls.get_user_chat_id(data)
        cls.update_chat_memory_cache(chat_id)
        return cls.chat_memory.load_memory(chat_id).get(INPUT_KEY)

    @classmethod
    def save_chat_memory(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        ChainLogger.debug("Saving chat memory", data)
        chat_id = cls.get_user_chat_id(data)
        cls.chat_memory.save_memory(chat_id, data)
        return cls.load_chat_memory(data) 

    @staticmethod
    def conversation_anonymized_if_sensitive_info(data: Dict[str, Any]) -> str:

        ChainLogger.info(
            f"Anoymizing conversation...: {data.get(ConversationActor.actor_key)}",
            data,
        )
        _extraction_list = data.get(MetadataActor.actor_key)['extraction_list']
        question = data.get(ConversationActor.actor_key)
        for item in _extraction_list:
            chunk = item['chunk']
            generalization = item['generalization']

            if item['sensitive']:
                words = generalization.split()
                semantic_words = "-".join(words)
                semantic_anonymization = semantic_words+"-"+str(uuid4())[:8]
                question = question.replace(chunk, semantic_anonymization) 
                item.update({"semantic_anonymization" : semantic_anonymization})
            else:
                item.update({"semantic_anonymization" : chunk})


        data.update({ConversationActor.actor_key: question})

        return data     

    @classmethod
    def retrieve_similar_data_from_question(cls, data: Dict[str, Any]) -> str:
        """
        Retrieval data from vectorstore (keyword + fuzzy) is added to the metadata actor to have a match between the anonymized data
        and the real data from de vectorstore.
        Using  keyword fuzzy search ["hormigón", "masa", "H20"] the retrieval is 
         ["HORMIGÓN EN MASA HE-20, DE CUALQUIER CONSISTENCIA Y TAMAÑO MÁXIMO DEL ÁRIDO 12-20 MM"] and then is tied up to uuid 
         4db0ea89-descripcionRecurso which is the anonymized data. At the sql execution time chunk 4db0ea89-descripcionRecurso
         will be substitute with the real data "HORMIGÓN EN MASA HE-20, DE CUALQUIER CONSISTENCIA Y TAMAÑO MÁXIMO DEL ÁRIDO 12-20 MM"

         At the same time, we add non sensitive records to a runnable variable that will be used in the sql actor prompt.
         If all is to be anonymized, then, there prompt will use an empty list 

        Example:
        {"extraction_list": [
            {'chunk': 'hormigon de basamento eh10', 
            'name': 'semantic_BDP.Recursos.descripcionRecurso', 
            'sensitive': True, 
            'chunk_reason': "'hormigon de basamento eh10' is a mix of words and...",
            'sensitive_reason': 'Table semantic_BDP.Recursos states-....',
            'generalization': 'material de construccion', 
            'generalization_reason': "material de construccion is a generalization.....",
            'semantic_anonymization': 'material-de-construccion-6a156a89', 
            'retrieval': ["HORMIGÓN EN MASA HE-20, DE CUALQUIER CONSISTENCIA Y TAMAÑO MÁXIMO DEL ÁRIDO 12-20 MM"]}
            ....
            ]
        }
        
        """
        context_question = data[ConversationActor.actor_key]
        _extraction_list = data[MetadataActor.actor_key]['extraction_list']

        index_name = cls.opensearch_vectorstore.index_name

        # Get the mapping of the index
        index_mapping = cls.opensearch_vectorstore.client.indices.get_mapping(index=index_name)

        # Extract column names from the mapping
        index_columns = index_mapping[index_name]['mappings']['properties'].keys()

        ChainLogger.info(
            f"{LOG_PREFIX} Retrieving data for question '{context_question}' ",
            data,
        )
   
        config_search = {"fuzziness":"AUTO",
                        "size" : 1, 
                        "search_type" : "match_search"}    

        retrieval_list: list = []

        for item in _extraction_list:
            
            chunk = item['chunk']
            text_field_in_opensearch = item['name'].split('.')[-1]

            # Solo buscamos en aquellos chunks asociados a una columna que exista en el indice en opensearch
            if text_field_in_opensearch in index_columns:
                # print(f"******* *******{text_field_in_opensearch}")
            
                retriever = cls.opensearch_vectorstore.as_retriever(**config_search,
                                                                    **{'chunk_text':chunk},
                                                                    **{'text_field':text_field_in_opensearch})
                
                retrieval = [reg.page_content for reg in (retriever.get_relevant_documents(query=context_question))]

                ChainLogger.info(
                    f"{LOG_PREFIX} Retrieval: {retrieval} for chunk '{chunk}' ",
                    data,
                )   

                item.update({"retrieval" : retrieval})

                if not item['sensitive']:
                    retrieval_list += [(text_field_in_opensearch + "=" + reg) for reg in retrieval]
        
        ChainLogger.info(
            f"{LOG_PREFIX} NOT sensitive data: {retrieval_list}",
            data,
        )        

        data.update({SIMILAR_DATA_RETRIEVAL: retrieval_list})

        return data  
      
    @classmethod
    def retrieve_similar_examples_from_question(cls, data: Dict[str, Any]) -> str:

        data_retrieval = data[SIMILAR_DATA_RETRIEVAL]

        examples_list = []

        data_collection = data[COLLECTION_INPUT_KEY]
        data = cls.get_examples_from_data_collection(data_collection)

        

        # examples_list = [SystemMessage(content="Here are some examples:\nEXAMPLES START")]
        for example in data:
           
            # bs_similar_examples = BeautifulSoup(example[SIMILAR_EXAMPLES], "html.parser").get_text()
            bs_similar_query = BeautifulSoup(example,"html.parser").get_text()
            
            # bs_similar_data_retrieval = BeautifulSoup(
            #  ",".join([c for c in example[SIMILAR_DATA_RETRIEVAL]]),
            #     "html.parser",
            # ).get_text()    


            provided_string_decoded = unidecode.unidecode(bs_similar_query)
            json_obj = json.loads(provided_string_decoded)

            id = json_obj["id"]
            query = json_obj["query"]
            similar_data_retrieval = json_obj[SIMILAR_DATA_RETRIEVAL]
            ai_response = json_obj['query_metadata']['sql_generation_list'][0]


            print(json.dumps(json_obj, indent=4))  # Optional: Dump the JSON object with indentation for readability

            # if len(bs_similar_data_retrieval) > 0:
            #     bs_similar_data_retrieval = f", '{bs_similar_data_retrieval}'"    

            example_response = json.dumps(example["query_metadata"])     
            bs_example_response = BeautifulSoup(example_response, "html.parser").get_text()

            
            # print(f"Similar data retrieval:{bs_similar_data_retrieval}")
            print(f"Similar example query:{bs_similar_query}")
            print(f"Example response:{bs_example_response}")
            
            _example = [
                # HumanMessage(
                #     content=SQLActorInput.create_input(
                #         **{TABLES_CONTEXT_KEY: example[TABLES_CONTEXT_KEY],
                #         SIMILAR_DATA_RETRIEVAL : bs_similar_data_retrieval,
                #         ConversationActor.actor_key: bs_similar_query})
                # ),
                # AIMessage(
                #     content=bs_example_response
                # )
            ]

            examples_list += _example

        
        ChainLogger.info(
            f"{LOG_PREFIX} EXAMPLES: {examples_list}",
            data,
        )        

        data.update({EXAMPLE_INJECTION: examples_list})

        return data 

    @classmethod
    def get_governance_examples_info(
        cls,
        collection_tables: List[str],
    ) -> List[GovernanceDataAsset]:
        """
        Obtain information from examples business data assets
        """
        try:
            if not cls.governance_service.governance_client:
                raise ValueError("Governance client is not initialized.")

            if not collection_tables:
                return []

            point_index = collection_tables[0].find(".")
            collection_name = (
                collection_tables[0][:point_index] if point_index != -1 else ""
            )

            if not collection_name:
                return []

            attribute_filter = OrderedDict(
                [
                    ("metadataPathLike", "glossary://BDP/Examples%"),
                    ("glossaryItemsTypeNames", "BUSINESS_ASSET"),
                ]
            )
            glossary_items_response = cls.governance_service.governance_client.get_glossary_items(
                attribute_filter
            )

            governance_data_tables = [
                # GovernanceDataAsset(name=glossaryItem.name,
                #                     descriptions=glossaryItem.description,
                #                     columns=[],)
                glossaryItem.description for glossaryItem in glossary_items_response.glossaryItems
            ]
            # return [doc for doc in governance_data_tables if doc is not None]
            return [doc for doc in governance_data_tables if doc is not None]
        except Exception as e:
            raise ValueError(
                f"GovernanceService: Error while retrieving Governance collection data: {e}"
            ) from e       


   
 


    @classmethod
    def chain(cls) -> Runnable:

        extractor_chain = MetadataActor(cls.openai_api_key, cls.llm_timeout).get_chain()
        conversation_chain = ConversationActor(cls.openai_api_key, cls.llm_timeout).get_chain()        

        composed_chain = (RunnablePassthrough() 
                            
            | RunnablePassthrough.assign(
                **{GENAI_AUTH_KEY: GenAiAuthRunnable()}) 
            # | RunnablePassthrough.assign(
            #     **{CHAT_ID_KEY: RunnableLambda(cls.get_or_create_chat_id)}) 
            # | RunnablePassthrough.assign( 
            #     **{ConversationActor.chat_history_key: RunnableLambda(cls.load_chat_memory)}) 

             
            # | RunnableLambda(lambda x: log_chain(x, "Starting conversation actor..."))
            # | RunnablePassthrough.assign(
            #     **{ConversationActor.actor_key: conversation_chain})    
            # | RunnablePassthrough.assign(
            #     **{ConversationActor.chat_history_key: RunnableLambda(cls.save_chat_memory)})  
            | RunnablePassthrough.assign(
                **{COLLECTION_TABLES_CONTEXT_KEY: RunnableLambda(cls.retrieve_collection_information)})  

            | RunnableLambda(lambda x: log_chain(x, "Starting extractor actor..."))
            | RunnablePassthrough.assign(
                **{MetadataActor.actor_key: extractor_chain})  

            | RunnablePassthrough(cls.conversation_anonymized_if_sensitive_info)  

            | RunnablePassthrough(lambda x: print(f"Conversation Actor.....: {x['conversation_actor']}"))

            | RunnablePassthrough(cls.retrieve_similar_data_from_question)
            | RunnablePassthrough(lambda x: print(f"Similar data.........: {x.get(SIMILAR_DATA_RETRIEVAL)}"))
            | RunnablePassthrough(lambda x: print(f"Metadata Actor.........: {x['metadata_actor']}"))
        )

        return composed_chain
    
    @classmethod
    def long_chain_with_sql(cls) -> Runnable:

        metadata_chain = MetadataActor(cls.openai_api_key, cls.llm_timeout).get_chain()
        conversation_chain = ConversationActor(cls.openai_api_key, cls.llm_timeout).get_chain()
        tables_chain = TablesActor(cls.openai_api_key, cls.llm_timeout).get_chain()
        
        sql_graph = SQLGraph(cls.openai_api_key,
                             cls.llm_timeout,
                             cls.virtualizer_service,
                             cls.opensearch_vectorstore,
                             graph_max_retries=3)
        
        sql_graph.sql_actor = SQLActor(cls.openai_api_key, cls.llm_timeout)
        # sql_graph.sql_actor = SQLActor(cls.openai_api_key, cls.llm_timeout,cls.governance_service, )
        explain_actor = ExplainActor(cls.openai_api_key, cls.llm_timeout)
        precision_actor = PrecisionActor(cls.openai_api_key, cls.llm_timeout)

        composed_chain = (RunnablePassthrough() 
                            
            | RunnablePassthrough.assign(
                **{GENAI_AUTH_KEY: GenAiAuthRunnable()}) 
            | RunnablePassthrough.assign(
                **{CHAT_ID_KEY: RunnableLambda(cls.get_or_create_chat_id)}) 
            | RunnablePassthrough.assign( 
                **{ConversationActor.chat_history_key: RunnableLambda(cls.load_chat_memory)}) 

             
            | RunnableLambda(lambda x: log_chain(x, "Starting conversation actor..."))
            | RunnablePassthrough.assign(
                **{ConversationActor.actor_key: conversation_chain})    
            | RunnablePassthrough(lambda x: print(f"1.########## Ouput Conversation Actor.....: {x['conversation_actor']}"))
            
            | RunnablePassthrough.assign(
                **{ConversationActor.chat_history_key: RunnableLambda(cls.save_chat_memory)})  
            | RunnablePassthrough.assign(
                **{COLLECTION_TABLES_CONTEXT_KEY: RunnableLambda(cls.retrieve_collection_information)})  

            | RunnableLambda(lambda x: log_chain(x, "Starting metadata actor..."))
            | RunnablePassthrough.assign(
                **{MetadataActor.actor_key: metadata_chain})  
            | RunnablePassthrough(lambda x: print(f"2.############## Output Metadata Actor.........: {x['metadata_actor']}"))

            | RunnablePassthrough(cls.conversation_anonymized_if_sensitive_info)  

            | RunnableLambda(lambda x: log_chain(x, "Starting tables actor..."))
            | RunnablePassthrough.assign(
                **{TablesActor.actor_key: tables_chain})   
            | RunnablePassthrough(lambda x: print(f"3.############## Output Tables Actor.........: {x['tables_actor']}"))

            | RunnablePassthrough.assign(
                **{REQUIRED_TABLES_KEY: RunnableLambda(cls.filter_required_tables)}) 
            | RunnablePassthrough.assign(
                **{TABLES_CONTEXT_KEY: RunnableLambda(cls.retrieve_tables_information)})                            

            | RunnablePassthrough(cls.retrieve_similar_data_from_question)
            | RunnablePassthrough(cls.retrieve_similar_examples_from_question)
            | RunnablePassthrough(sql_graph.run_multiple_sql_chain)
            | RunnablePassthrough(lambda x: print(f"4.############### Output Actor data.........: {x.get(SQLActor.actor_key)}"))

            | RunnableLambda(lambda x: log_chain(x, "[SQL_GRAPH] Starting explain and precision actor"))
            | RunnablePassthrough.assign(
                **{ExplainActor.actor_key: explain_actor.get_chain()}
            )
            | RunnablePassthrough(lambda x: print(f".5 ############## Output Explain actor.........: {x.get(ExplainActor.actor_key)}"))
            | RunnablePassthrough.assign(
                **{PrecisionActor.actor_key: precision_actor.get_chain()}
            )            
            | RunnablePassthrough(lambda x: print(f"6. ############## Output Precision actor.........: {x.get(PrecisionActor.actor_key)}"))
            | RunnablePassthrough(sql_graph.validate_precision_actor)


        )

        return composed_chain.with_config(config=RunnableConfig(recursion_limit=200))    



def _test_chain():
    
    config = RunnableConfig(metadata={"x": 1})
    generic_chain = GenericChain(opensearch_index="adif-materials-nlp-index-analyzer")
    chain = generic_chain.chain()

    result = chain.invoke(
        {
            INPUT_KEY : "",
            # ConversationActor.actor_key : "Dame el precio del hormigon en masa h20 con codigo ZZC5345",
            # ConversationActor.actor_key : "Cual es el precio, unidad y nombre de los recursos que se utilizan para los andamios cimbra",
            # ConversationActor.actor_key : "Dame todos los recursos de la categoría con codigo MT303#",
            # ConversationActor.actor_key : "Dame el precio del hormigon de basamento eh10 con codigo B99 y la unidad de una biga artesa de valor 100",
            # ConversationActor.actor_key : "Dime la descripcion de los elementos necesarios para construir andamios cimbra",

            # ConversationActor.actor_key : "Dame el precio de la viga artesa de 90 con codigo ZZC5345",
            # ConversationActor.actor_key : "Dame el precio del hormigon en masa h20 con codigo AU1 y el precio de la viga artesa",
            # ConversationActor.actor_key : "Dime que puedo construir con garbancillo de 20mm",
            # ConversationActor.actor_key : "Dime la descripcion de los elementos necesarios para construir andamios cimbra",
            # ConversationActor.actor_key : "dame el precio y descripcion de los recursos del convenio del metal",
            ConversationActor.actor_key : "Cuál es el precio del transporte de tierras con camión, para las obras de Adif AV",
            COLLECTION_INPUT_KEY: "semantic_BDP",
        },
        config=config,
    )

def _test_long_chain():
    
    config = RunnableConfig(metadata={"x": 1})
    generic_chain = GenericChain(opensearch_index="adif-materials-nlp-index-analyzer")
    chain = generic_chain.long_chain_with_sql()

    result = chain.invoke(
        {
            INPUT_KEY : "Dame el precio del hormigon en masa h20 con codigo ZZC5345",
            # INPUT_KEY : "Dame la descripción, unidad, precio y código de la viga prefabricada",
            # INPUT_KEY : "Dime que puedo construir con garbancillo de 20mm",
            # INPUT_KEY : "Dame la descripción, unidad, precio y código de los andamios cimbra", # no hay datos
            # INPUT_KEY : "Cual es el precio, unidad y nombre de los recursos que se utilizan para los andamios cimbra",
            # INPUT_KEY : "dame el precio y descripcion de los recursos del convenio del metal",
            # INPUT_KEY : "Dame todos los recursos de la categoría con codigo MT303#",
            # INPUT_KEY : "Dame el precio del hormigon de basamento eh10 con codigo B99 y la unidad de una biga artesa de valor 100",
            # INPUT_KEY : "Dime la descripcion de los elementos necesarios para construir andamios cimbra",
            # INPUT_KEY : "Dame el precio de la viga artesa de 90 con codigo ZZC5345",
            # INPUT_KEY : "Dame el precio del hormigon en masa h20 con codigo AU1 y el precio de la viga artesa",
            # INPUT_KEY : "Dime la descripcion de los elementos necesarios para construir andamios cimbra",
            COLLECTION_INPUT_KEY: "semantic_BDP",
        },
        config=config,
    )


if __name__ == "__main__":
    _test_long_chain()
    # _test_chain()