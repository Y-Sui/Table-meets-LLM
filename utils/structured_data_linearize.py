class StructuredDataLinearize:
    """Expects the structured data with the following format:

        {
            'title': title of the form,
            'description': description of the form,
            'body': [
                # a list of blocks
                {
                    'type': type of the block,
                    'title': title of the block,
                    'description': description of the block
                    'options':
                        # For choice type and rating type
                        [a list of option]
                        # For likert type
                        {
                            'rows': [a list of row captions]
                            'columns': [a list of column captions]
                        }
                }
            ]
        }
        """
    def __init__(self):

        pass

