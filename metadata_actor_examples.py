
from genai_chain_sql.constants import COLLECTION_TABLES_CONTEXT_KEY

TABLE_CONTEXT = """[Table(learning.customer, description:'Almacena información de los clientes, incluyendo datos personales y de contacto. La columna name almacena informacion sensible', columns:[('customer_id', INTEGER), ('name', TEXT)]) 
                                            Table(learning.product, description:'Contiene detalles sobre los productos ofrecidos, incluyendo precios y categorías.', columns:[('product_id', TEXT), ('name', TEXT), ('price', DOUBLE)]) """


_V4_METADATA_ACTOR_EXAMPLES = [
    {
        "id": 1,
        "query": "Dame el identificador del usuario Alvaro Jimenez" ,
        COLLECTION_TABLES_CONTEXT_KEY : TABLE_CONTEXT,
        "query_metadata": { "extraction_list" : [ { "chunk": "Alvaro Jimenez",
                                                    "name" : "learning.customer.name", 
                                                    "sensitive": True,
                                                    "chunk_reason": "'Alvaro Jimenez' is a group of words that can be used in a WHERE clause like this: learning.customer.name = 'Alvaro Jimenez'",
                                                    "sensitive_reason": "Table learning.customer states in its description that learning.customer.name has sensitive data",
                                                    "generalization" : "identificador",
                                                    "generalization_reason" : "identificador is a generalization for the group of words 'Alvaro Jimenez'"
                                                    }
                                                ]
                        }                    
    },
    {
        "id": 2,
        "query": "Dame el valor del alcohol de ricino de 96 grados con identificador z77j que vale mas de 10 euros" ,
        COLLECTION_TABLES_CONTEXT_KEY : TABLE_CONTEXT,
        "query_metadata": { "extraction_list" : [ { "chunk": "alcohol de ricino de 96 grados",
                                                    "name" : "learning.product.name", 
                                                    "sensitive": False,
                                                    "chunk_reason": "'alcohol de ricino de 96 grados' is a mix of words and numbers that can be used in a WHERE clause like this: learning.product.name = 'alcohol de ricino de 96 grados'",
                                                    "sensitive_reason": "Table learning.product has no info in its description about the presence of sensitive data",
                                                    "generalization" : "liquido", 
                                                    "generalization_reason" : "liquido is a generalization for the group of words 'alcohol de ricion de 96 grados'"
                                                    },
                                                     { "chunk": "10",
                                                    "name" : "learning.product.price", 
                                                    "sensitive": False,
                                                    "chunk_reason": "'10' is a number that best fits in a WHERE clause like this: learning.product.price = '10'",
                                                    "sensitive_reason": "Table learning.product has no info in its description about the presence of sensitive data",
                                                    "generalization" : "valor",
                                                    "generalization_reason" : "valor is a generalization for the group of words '10'"
                                                    },
                                                     { "chunk": "z77j",
                                                    "name" : "learning.product.product_id", 
                                                    "sensitive": False,
                                                    "chunk_reason": "'z77j' is a code that can be used in a WHERE clause like this: learning.product.product_id = 'z77j'",
                                                    "sensitive_reason": "Table learning.product has no info in its description about the presence of sensitive data",
                                                    "generalization" : "identificador",
                                                    "generalization_reason" : "identificador is a generalization for the group of words '10'"
                                                    }
                                                ]
                        }                    
    }, 
    {
        "id": 3,
        "query": "Dame el precio, descripcion y unidad del ladriyo perfforado de 59 puntas" ,
        COLLECTION_TABLES_CONTEXT_KEY : TABLE_CONTEXT,
        "query_metadata": { "extraction_list" : [ { "chunk": "ladriyo perfforado de 59 puntas",
                                                    "name" : "learning.product.name", 
                                                    "sensitive": False,
                                                    "chunk_reason": "'ladriyo perfforado de 59 puntas' is a mix of words and numbers that can be used in a WHERE clause like this: learning.product.name = 'alcohol de ricino de 96 grados'",
                                                    "sensitive_reason": "Table learning.product has no info in its description about the presence of sensitive data",
                                                    "generalization" : "material",
                                                    "generalization_reason" : "material is a generalization for the group of words '10'"
                                                    }
                                                ]
                        }                    
    } 
]
