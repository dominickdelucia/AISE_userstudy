# coding: utf-8

from flask import Blueprint
from flask_apispec import marshal_with
from flask_jwt_extended import current_user, jwt_required, jwt_optional

from conduit.exceptions import InvalidUsage
from conduit.user.models import User
from .serializers import profile_schema

blueprint = Blueprint('profiles', __name__)


@blueprint.route('/api/profiles/<username>', methods=('GET',))
@jwt_optional
@marshal_with(profile_schema)
def get_profile(username):
    user = User.query.